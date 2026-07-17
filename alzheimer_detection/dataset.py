"""
dataset.py — PyTorch Dataset for multi-modal Alzheimer's detection.

Returns dict: {
    'mri':      FloatTensor (1, D, H, W),
    'clinical': FloatTensor (CLINICAL_DIM,),
    'label':    LongTensor  ()
}
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

from config import (
    PROCESSED_DIR, CLINICAL_CSV, CLASSES, CLASS_TO_IDX,
    CLINICAL_FEATURES, CLINICAL_DIM, IMG_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED,
    AUG_FLIP_PROB, AUG_ROTATE_DEGREES, AUG_NOISE_STD,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
)


# ─── Clinical data loader ─────────────────────────────────────────────────────

def load_clinical_data(csv_path: str) -> tuple[pd.DataFrame, StandardScaler]:
    """Load and normalise clinical features from ADNIMERGE.csv."""
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)

    # Map DX to our 3 classes
    dx_map = {
        'CN': 'CN', 'SMC': 'CN',
        'EMCI': 'MCI', 'LMCI': 'MCI', 'MCI': 'MCI',
        'AD': 'AD', 'Dementia': 'AD'
    }
    df['label'] = df['DX'].map(dx_map)
    df = df[df['label'].isin(CLASSES)].copy()

    # Keep only needed columns
    available = [f for f in CLINICAL_FEATURES if f in df.columns]
    df = df[['PTID', 'VISCODE', 'label'] + available].copy()

    # Fill missing with median
    for col in available:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    # Encode gender
    if 'PTGENDER' in df.columns:
        df['PTGENDER'] = (df['PTGENDER'] == 'Male').astype(float)

    scaler = StandardScaler()
    df[available] = scaler.fit_transform(df[available])

    return df, scaler


# ─── Augmentation ─────────────────────────────────────────────────────────────

def random_flip(volume: np.ndarray) -> np.ndarray:
    """Random left-right flip."""
    if random.random() < AUG_FLIP_PROB:
        volume = np.flip(volume, axis=2).copy()
    return volume


def random_noise(volume: np.ndarray) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, AUG_NOISE_STD, volume.shape).astype(np.float32)
    return volume + noise


def random_rotate90(volume: np.ndarray) -> np.ndarray:
    """Random 90-degree rotation in axial plane (fast, no interpolation)."""
    k = random.randint(0, 3)
    return np.rot90(volume, k, axes=(1, 2)).copy()


def augment(volume: np.ndarray) -> np.ndarray:
    volume = random_flip(volume)
    volume = random_rotate90(volume)
    volume = random_noise(volume)
    return volume


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AlzheimerDataset(Dataset):
    def __init__(self, samples: list, clinical_df: pd.DataFrame | None,
                 is_train: bool = False):
        """
        Args:
            samples: list of (npy_path, label_idx, subject_id)
            clinical_df: preprocessed clinical dataframe or None
            is_train: whether to apply augmentation
        """
        self.samples     = samples
        self.clinical_df = clinical_df
        self.is_train    = is_train

    def __len__(self):
        return len(self.samples)

    def _get_clinical(self, subject_id: str) -> torch.Tensor:
        """Look up clinical features for this subject."""
        if self.clinical_df is None:
            return torch.zeros(CLINICAL_DIM, dtype=torch.float32)

        row = self.clinical_df[self.clinical_df['PTID'] == subject_id]
        if row.empty:
            return torch.zeros(CLINICAL_DIM, dtype=torch.float32)

        # Use most recent visit
        row = row.iloc[-1]
        available = [f for f in CLINICAL_FEATURES if f in row.index]
        vec = np.zeros(CLINICAL_DIM, dtype=np.float32)
        for i, feat in enumerate(CLINICAL_FEATURES):
            if feat in row.index:
                vec[i] = float(row[feat])
        return torch.tensor(vec, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        npy_path, label_idx, subject_id = self.samples[idx]

        # Load MRI volume
        volume = np.load(npy_path).astype(np.float32)

        if self.is_train:
            volume = augment(volume)

        # Add channel dim: (1, D, H, W)
        mri_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

        return {
            'mri':      mri_tensor,
            'clinical': self._get_clinical(subject_id),
            'label':    torch.tensor(label_idx, dtype=torch.long),
            'subject':  subject_id,
            'path':     npy_path
        }


# ─── Build sample list ────────────────────────────────────────────────────────

def build_samples(processed_dir: str) -> list:
    """Walk PROCESSED_DIR and collect (path, label, subject_id) tuples."""
    samples = []
    for cls in CLASSES:
        cls_dir = os.path.join(processed_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        label_idx = CLASS_TO_IDX[cls]
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.npy'):
                continue
            # Extract subject ID from ADNI filename convention
            subject_id = fname.split('_')[0] if '_' in fname else fname[:-4]
            samples.append((
                os.path.join(cls_dir, fname),
                label_idx,
                subject_id
            ))
    return samples


def get_class_weights(samples: list) -> torch.Tensor:
    """Inverse-frequency class weights for WeightedRandomSampler."""
    counts = np.zeros(len(CLASSES))
    for _, label_idx, _ in samples:
        counts[label_idx] += 1
    weights = 1.0 / (counts + 1e-6)
    sample_weights = torch.tensor(
        [weights[label_idx] for _, label_idx, _ in samples],
        dtype=torch.float32
    )
    return sample_weights


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(processed_dir: str = PROCESSED_DIR,
                    clinical_csv: str = CLINICAL_CSV):
    """
    Returns:
        train_loader, val_loader, test_loader, class_weights (for loss)
    """
    samples = build_samples(processed_dir)
    if not samples:
        raise FileNotFoundError(
            f"No .npy files found in {processed_dir}. "
            "Run preprocessing.py first, or check your data folder structure."
        )

    print(f"Total samples found: {len(samples)}")
    for cls in CLASSES:
        n = sum(1 for _, l, _ in samples if l == CLASS_TO_IDX[cls])
        print(f"  {cls}: {n}")

    clinical_df, _ = load_clinical_data(clinical_csv)

    # Stratified split
    indices = list(range(len(samples)))
    labels  = [s[1] for s in samples]

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - TRAIN_SPLIT),
        stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio),
        stratify=temp_labels, random_state=SEED
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    test_samples  = [samples[i] for i in test_idx]

    print(f"\nSplit — Train: {len(train_samples)} | "
          f"Val: {len(val_samples)} | Test: {len(test_samples)}")

    # Weighted sampler for class imbalance
    sample_weights = get_class_weights(train_samples)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_samples), replacement=True
    )

    train_loader = DataLoader(
        AlzheimerDataset(train_samples, clinical_df, is_train=True),
        batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        AlzheimerDataset(val_samples, clinical_df, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        AlzheimerDataset(test_samples, clinical_df, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Class weights tensor for weighted cross-entropy
    counts = np.zeros(len(CLASSES))
    for _, l, _ in train_samples:
        counts[l] += 1
    class_weights = torch.tensor(
        counts.sum() / (len(CLASSES) * counts + 1e-6),
        dtype=torch.float32
    )

    return train_loader, val_loader, test_loader, class_weights
