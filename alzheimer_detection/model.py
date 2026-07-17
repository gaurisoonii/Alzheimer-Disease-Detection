
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_CLASSES, CLINICAL_DIM,
    MRI_FEATURE_DIM, CLINICAL_HIDDEN, FUSION_HIDDEN, DROPOUT
)


# ─── 3D Residual Block ────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ─── 3D CNN Branch ────────────────────────────────────────────────────────────

class MRIEncoder(nn.Module):
    """
    Lightweight 3D ResNet for volumetric MRI.
    Input:  (B, 1, 96, 96, 96)
    Output: (B, MRI_FEATURE_DIM)
    """
    def __init__(self, feature_dim: int = MRI_FEATURE_DIM):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(ResBlock3D(32,  64,  stride=1),
                                    ResBlock3D(64,  64,  stride=1))
        self.layer2 = nn.Sequential(ResBlock3D(64,  128, stride=2),
                                    ResBlock3D(128, 128, stride=1))
        self.layer3 = nn.Sequential(ResBlock3D(128, 256, stride=2),
                                    ResBlock3D(256, 256, stride=1))

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout      = nn.Dropout(p=DROPOUT)
        self.fc           = nn.Linear(256, feature_dim)
        self.bn_out       = nn.BatchNorm1d(feature_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_out(x)
        return F.relu(x)

    def get_cam_target_layer(self):
        """Return the last conv layer for Grad-CAM."""
        return self.layer3[-1].conv2


# ─── Clinical MLP Branch ──────────────────────────────────────────────────────

class ClinicalEncoder(nn.Module):
    """
    MLP for tabular clinical features.
    Input:  (B, CLINICAL_DIM)
    Output: (B, CLINICAL_HIDDEN)
    """
    def __init__(self, input_dim: int = CLINICAL_DIM,
                 hidden_dim: int = CLINICAL_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Fusion Classifier ────────────────────────────────────────────────────────

class MultiModalFusionModel(nn.Module):
    """
    Late-fusion model combining MRI (3D CNN) + clinical (MLP) branches.

    Attributes:
        mri_encoder:       3D ResNet backbone
        clinical_encoder:  Tabular MLP
        fusion:            Post-concat MLP + classifier
    """
    def __init__(self):
        super().__init__()
        self.mri_encoder      = MRIEncoder(feature_dim=MRI_FEATURE_DIM)
        self.clinical_encoder = ClinicalEncoder(input_dim=CLINICAL_DIM,
                                                hidden_dim=CLINICAL_HIDDEN)

        concat_dim = MRI_FEATURE_DIM + CLINICAL_HIDDEN

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, FUSION_HIDDEN),
            nn.BatchNorm1d(FUSION_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(FUSION_HIDDEN, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT * 0.5),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, mri: torch.Tensor,
                clinical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mri:      (B, 1, D, H, W)
            clinical: (B, CLINICAL_DIM)
        Returns:
            logits:   (B, NUM_CLASSES)
        """
        mri_feat      = self.mri_encoder(mri)
        clinical_feat = self.clinical_encoder(clinical)
        fused         = torch.cat([mri_feat, clinical_feat], dim=1)
        features      = self.fusion(fused)
        return self.classifier(features)

    def get_cam_target_layer(self):
        return self.mri_encoder.get_cam_target_layer()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Model summary ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MultiModalFusionModel()
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    dummy_mri      = torch.randn(2, 1, 96, 96, 96)
    dummy_clinical = torch.randn(2, CLINICAL_DIM)
    logits = model(dummy_mri, dummy_clinical)
    print(f"Output shape: {logits.shape}")   # (2, 3)
