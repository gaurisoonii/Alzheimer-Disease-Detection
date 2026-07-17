"""
preprocessing.py — MRI preprocessing pipeline.

Steps:
  1. Load NIFTI volume
  2. Resample to isotropic voxel spacing
  3. Skull strip (via nilearn masking)
  4. Intensity normalisation (z-score)
  5. Crop / pad to target size
  6. Save as .npy for fast loading during training
"""

import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import (
    RAW_DIR, PROCESSED_DIR, CLASSES,
    IMG_SIZE, VOXEL_SPACING
)


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIFTI file and return (data, affine)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def resample_volume(data: np.ndarray, affine: np.ndarray,
                    target_spacing: tuple) -> np.ndarray:
    """Resample volume to target voxel spacing."""
    current_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    scale_factors = current_spacing / np.array(target_spacing)
    resampled = zoom(data, scale_factors, order=1)
    return resampled


def skull_strip(data: np.ndarray) -> np.ndarray:
    """Simple intensity-based skull stripping (threshold + largest component)."""
    from scipy.ndimage import label, binary_fill_holes, binary_erosion

    threshold = np.percentile(data[data > 0], 15)
    mask = data > threshold
    mask = binary_fill_holes(mask)
    labeled, num = label(mask)
    if num == 0:
        return data
    sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1
    brain_mask = labeled == largest
    brain_mask = binary_fill_holes(brain_mask)
    return data * brain_mask


def normalise(data: np.ndarray) -> np.ndarray:
    """Z-score normalisation on brain voxels only."""
    brain = data[data > 0]
    if brain.size == 0:
        return data
    mean, std = brain.mean(), brain.std()
    if std < 1e-6:
        return data
    return (data - mean) / std


def crop_or_pad(data: np.ndarray, target: tuple) -> np.ndarray:
    """Centre-crop or zero-pad each dimension to target size."""
    result = np.zeros(target, dtype=np.float32)
    for dim in range(3):
        src_size = data.shape[dim]
        tgt_size = target[dim]
        src_start = max(0, (src_size - tgt_size) // 2)
        tgt_start = max(0, (tgt_size - src_size) // 2)
        src_end = src_start + min(src_size, tgt_size)
        tgt_end = tgt_start + min(src_size, tgt_size)

    # Apply slicing along all dims simultaneously
    slices_src, slices_tgt = [], []
    for dim in range(3):
        src_size = data.shape[dim]
        tgt_size = target[dim]
        src_start = max(0, (src_size - tgt_size) // 2)
        tgt_start = max(0, (tgt_size - src_size) // 2)
        length = min(src_size - src_start, tgt_size - tgt_start)
        slices_src.append(slice(src_start, src_start + length))
        slices_tgt.append(slice(tgt_start, tgt_start + length))

    result[tuple(slices_tgt)] = data[tuple(slices_src)]
    return result


def preprocess_one(nifti_path: str, save_path: str) -> bool:
    """Full pipeline for a single scan. Returns True on success."""
    try:
        data, affine = load_nifti(nifti_path)
        data = resample_volume(data, affine, VOXEL_SPACING)
        data = skull_strip(data)
        data = normalise(data)
        data = crop_or_pad(data, IMG_SIZE)
        np.save(save_path, data)
        return True
    except Exception as e:
        print(f"  ERROR processing {nifti_path}: {e}")
        return False


def preprocess_dataset():
    """Preprocess all classes and save to PROCESSED_DIR."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stats = {"success": 0, "failed": 0}

    for cls in CLASSES:
        raw_cls_dir = os.path.join(RAW_DIR, cls)
        proc_cls_dir = os.path.join(PROCESSED_DIR, cls)
        os.makedirs(proc_cls_dir, exist_ok=True)

        if not os.path.exists(raw_cls_dir):
            print(f"  Skipping {cls} — folder not found at {raw_cls_dir}")
            continue

        files = [f for f in os.listdir(raw_cls_dir)
                 if f.endswith(('.nii', '.nii.gz'))]
        print(f"\nProcessing {cls}: {len(files)} scans")

        for fname in tqdm(files, desc=cls):
            src = os.path.join(raw_cls_dir, fname)
            dst = os.path.join(proc_cls_dir,
                               fname.replace('.nii.gz', '.npy')
                                    .replace('.nii', '.npy'))
            if os.path.exists(dst):
                stats["success"] += 1
                continue
            ok = preprocess_one(src, dst)
            stats["success" if ok else "failed"] += 1

    print(f"\nPreprocessing complete. "
          f"Success: {stats['success']}  Failed: {stats['failed']}")


if __name__ == "__main__":
    preprocess_dataset()
