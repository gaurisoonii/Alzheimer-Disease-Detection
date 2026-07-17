"""
gradcam.py — Grad-CAM for 3D MRI Alzheimer's detection.

Produces:
  - 3D activation heatmap (same shape as input volume)
  - Axial / sagittal / coronal 2D slice overlays
  - Per-class saliency maps showing which brain regions drove prediction
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom

from config import (
    BEST_MODEL_PATH, RESULTS_DIR, CLASSES, IDX_TO_CLASS,
    IMG_SIZE, NUM_CLASSES
)
from model import MultiModalFusionModel


# ─── Grad-CAM 3D ──────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D CNNs.

    Usage:
        gcam = GradCAM3D(model)
        cam, pred_class, confidence = gcam(mri_tensor, clinical_tensor)
        gcam.remove_hooks()
    """

    def __init__(self, model: MultiModalFusionModel):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.get_cam_target_layer()

        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, mri: torch.Tensor, clinical: torch.Tensor,
                 target_class: int | None = None):
        """
        Args:
            mri:          (1, 1, D, H, W) — single scan
            clinical:     (1, CLINICAL_DIM)
            target_class: if None, uses predicted class

        Returns:
            cam:          np.ndarray (D, H, W) — normalised heatmap [0,1]
            pred_class:   int
            confidence:   float
        """
        self.model.eval()
        mri      = mri.requires_grad_(True)
        logits   = self.model(mri, clinical)
        probs    = torch.softmax(logits, dim=1)[0]

        pred_class  = logits.argmax(dim=1).item()
        confidence  = probs[pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backprop to target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Pool gradients over spatial dims → channel weights
        grads = self.gradients[0]          # (C, D, H, W)
        acts  = self.activations[0]        # (C, D, H, W)
        weights = grads.mean(dim=(1, 2, 3))  # (C,)

        # Weighted sum of activation maps
        cam = (weights[:, None, None, None] * acts).sum(dim=0)  # (D, H, W)
        cam = F.relu(cam)

        # Upsample to original MRI size
        cam_np = cam.cpu().numpy()
        if cam_np.shape != IMG_SIZE:
            scale = tuple(t / s for t, s in zip(IMG_SIZE, cam_np.shape))
            cam_np = zoom(cam_np, scale, order=1)

        # Normalise to [0, 1]
        cam_np -= cam_np.min()
        if cam_np.max() > 1e-8:
            cam_np /= cam_np.max()

        return cam_np, pred_class, confidence


# ─── Visualisation ────────────────────────────────────────────────────────────

def overlay_cam_on_slice(mri_slice: np.ndarray,
                          cam_slice: np.ndarray,
                          alpha: float = 0.45) -> np.ndarray:
    """Blend greyscale MRI slice with coloured CAM heatmap."""
    # Normalise MRI slice to [0,1]
    s = mri_slice.copy().astype(np.float32)
    s -= s.min()
    if s.max() > 1e-8:
        s /= s.max()

    # Convert MRI to RGB greyscale
    mri_rgb = np.stack([s, s, s], axis=-1)

    # Map CAM to colour
    colormap = cm.get_cmap('jet')
    cam_rgb  = colormap(cam_slice)[:, :, :3].astype(np.float32)

    # Blend
    blended = (1 - alpha) * mri_rgb + alpha * cam_rgb
    return np.clip(blended, 0, 1)


def visualise_gradcam(mri_volume: np.ndarray, cam_volume: np.ndarray,
                       pred_class: int, confidence: float,
                       true_class: int | None = None,
                       save_path: str = None) -> plt.Figure:
    """
    Plot axial / sagittal / coronal central slices with CAM overlay.

    Args:
        mri_volume: (D, H, W) normalised MRI
        cam_volume: (D, H, W) Grad-CAM heatmap in [0,1]
        pred_class: predicted class index
        confidence: prediction confidence
        true_class: ground truth class index (optional)
        save_path:  if given, saves figure here
    """
    D, H, W = mri_volume.shape
    slices = {
        'axial':     (mri_volume[D//2, :, :],  cam_volume[D//2, :, :]),
        'coronal':   (mri_volume[:, H//2, :],  cam_volume[:, H//2, :]),
        'sagittal':  (mri_volume[:, :, W//2],  cam_volume[:, :, W//2]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')

    pred_label = CLASSES[pred_class]
    true_label = CLASSES[true_class] if true_class is not None else "?"
    correct    = (true_class == pred_class) if true_class is not None else None

    title_color = '#66ff66' if correct else ('#ff6666' if correct is False else '#ffffff')
    suptitle = (f"Prediction: {pred_label} ({confidence*100:.1f}%)"
                + (f"  |  Ground truth: {true_label}" if true_class is not None else ""))
    fig.suptitle(suptitle, color=title_color, fontsize=14, fontweight='bold')

    for ax, (plane, (mri_sl, cam_sl)) in zip(axes, slices.items()):
        blended = overlay_cam_on_slice(mri_sl, cam_sl)
        ax.imshow(blended, aspect='auto')
        ax.set_title(plane.capitalize(), color='white', fontsize=11)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())

    return fig


def visualise_all_classes(mri_volume: np.ndarray, clinical: torch.Tensor,
                           model: MultiModalFusionModel, device,
                           save_path: str = None) -> plt.Figure:
    """
    Generate Grad-CAM for each class side-by-side (axial slice only).
    Useful for showing which regions matter for CN vs MCI vs AD.
    """
    gcam = GradCAM3D(model)
    D    = mri_volume.shape[0]
    mri_tensor = torch.tensor(mri_volume, dtype=torch.float32
                              ).unsqueeze(0).unsqueeze(0).to(device)
    clinical   = clinical.unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 5))
    fig.patch.set_facecolor('#1a1a1a')

    for i, cls in enumerate(CLASSES):
        cam, pred_class, conf = gcam(mri_tensor.clone(), clinical.clone(),
                                     target_class=i)
        mri_sl = mri_volume[D // 2, :, :]
        cam_sl = cam[D // 2, :, :]
        blended = overlay_cam_on_slice(mri_sl, cam_sl)

        axes[i].imshow(blended, aspect='auto')
        marker = ' ◀' if i == pred_class else ''
        axes[i].set_title(f'{cls} CAM{marker}', color='white', fontsize=12)
        axes[i].axis('off')

    gcam.remove_hooks()
    plt.suptitle('Grad-CAM per class (axial slice)',
                 color='white', fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    return fig


# ─── Inference + CAM on a single scan ─────────────────────────────────────────

def predict_with_gradcam(npy_path: str,
                          clinical_vec: np.ndarray | None = None,
                          model_path: str = BEST_MODEL_PATH,
                          save_dir: str = RESULTS_DIR):
    """
    Load a preprocessed .npy MRI, run inference, and generate Grad-CAM.

    Args:
        npy_path:     path to .npy volume
        clinical_vec: np.array of shape (CLINICAL_DIM,) or None
        model_path:   path to saved checkpoint
        save_dir:     where to save output images

    Returns:
        pred_class, confidence, cam_volume
    """
    from config import CLINICAL_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MultiModalFusionModel().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Prepare inputs
    volume    = np.load(npy_path).astype(np.float32)
    mri_t     = torch.tensor(volume).unsqueeze(0).unsqueeze(0).to(device)
    if clinical_vec is None:
        clinical_vec = np.zeros(CLINICAL_DIM, dtype=np.float32)
    clinical_t = torch.tensor(clinical_vec, dtype=torch.float32
                               ).unsqueeze(0).to(device)

    # Grad-CAM
    gcam = GradCAM3D(model)
    cam, pred_class, confidence = gcam(mri_t, clinical_t)
    gcam.remove_hooks()

    # Save visualisations
    os.makedirs(save_dir, exist_ok=True)
    subject = os.path.basename(npy_path).replace('.npy', '')

    visualise_gradcam(
        volume, cam, pred_class, confidence,
        save_path=os.path.join(save_dir, f'{subject}_gradcam.png')
    )
    visualise_all_classes(
        volume, torch.tensor(clinical_vec, dtype=torch.float32),
        model, device,
        save_path=os.path.join(save_dir, f'{subject}_gradcam_all_classes.png')
    )

    print(f"Prediction: {CLASSES[pred_class]} ({confidence*100:.1f}%)")
    print(f"Saved Grad-CAM to {save_dir}/")
    return pred_class, confidence, cam


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to .npy MRI file')
    parser.add_argument('--model', default=BEST_MODEL_PATH)
    parser.add_argument('--out',   default=RESULTS_DIR)
    args = parser.parse_args()
    predict_with_gradcam(args.npy, model_path=args.model, save_dir=args.out)
