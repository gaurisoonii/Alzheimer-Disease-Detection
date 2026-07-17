"""
config.py — Central configuration for all hyperparameters and paths.
Change values here; everything else reads from this file.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
CLINICAL_CSV    = os.path.join(DATA_DIR, "clinical", "ADNIMERGE.csv")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASSES         = ["CN", "MCI", "AD"]
NUM_CLASSES     = len(CLASSES)
CLASS_TO_IDX    = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS    = {i: c for c, i in CLASS_TO_IDX.items()}

# ─── MRI Volume Settings ──────────────────────────────────────────────────────
IMG_SIZE        = (96, 96, 96)          # (D, H, W) after resampling
VOXEL_SPACING   = (1.5, 1.5, 1.5)      # mm isotropic

# ─── Clinical Features ────────────────────────────────────────────────────────
CLINICAL_FEATURES = [
    "AGE", "PTGENDER", "PTEDUCAT",
    "MMSE", "CDRSB", "ADAS11",
    "Hippocampus", "Entorhinal",
    "APOE4"
]
CLINICAL_DIM    = len(CLINICAL_FEATURES)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 4                     # keep low for 3D volumes
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
LR_PATIENCE     = 5                     # ReduceLROnPlateau patience
EARLY_STOP      = 10                    # early stopping patience
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
NUM_WORKERS     = 4
PIN_MEMORY      = True
SEED            = 42

# ─── Model ────────────────────────────────────────────────────────────────────
MRI_FEATURE_DIM     = 256               # output dim of 3D CNN branch
CLINICAL_HIDDEN     = 128               # hidden dim of clinical MLP
FUSION_HIDDEN       = 256               # fusion layer hidden dim
DROPOUT             = 0.4

# ─── Checkpoint ───────────────────────────────────────────────────────────────
BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUG_FLIP_PROB       = 0.5
AUG_ROTATE_DEGREES  = 10
AUG_NOISE_STD       = 0.02
