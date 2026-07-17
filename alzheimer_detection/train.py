"""
train.py — Full training loop with:
  - Weighted cross-entropy for class imbalance
  - ReduceLROnPlateau scheduler
  - Model checkpointing (best val AUC + last epoch)
  - Early stopping
  - TensorBoard logging
  - Full metrics per epoch (loss, acc, AUC, F1)
  - Confusion matrix + classification report saved to results/
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    roc_auc_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR,
    BEST_MODEL_PATH, LAST_MODEL_PATH,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, EARLY_STOP, SEED,
    NUM_CLASSES, CLASSES, IDX_TO_CLASS
)
from model import MultiModalFusionModel
from dataset import get_dataloaders


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class='ovr', average='macro'
        )
    except Exception:
        auc = 0.0
    return acc, f1, auc


# ─── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if is_train else "Val",
                          leave=False):
            mri      = batch['mri'].to(device, non_blocking=True)
            clinical = batch['clinical'].to(device, non_blocking=True)
            labels   = batch['label'].to(device, non_blocking=True)

            logits = model(mri, clinical)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            total_loss  += loss.item() * mri.size(0)
            all_labels  += labels.cpu().numpy().tolist()
            all_preds   += preds.tolist()
            all_probs   += probs.tolist()

    avg_loss        = total_loss / len(loader.dataset)
    acc, f1, auc    = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, acc, f1, auc, all_labels, all_preds


# ─── Confusion matrix plot ────────────────────────────────────────────────────

def save_confusion_matrix(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─── Training curve plot ──────────────────────────────────────────────────────

def save_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ['loss', 'acc', 'auc']):
        ax.plot(history[f'train_{key}'], label='train')
        ax.plot(history[f'val_{key}'],   label='val')
        ax.set_title(key.upper())
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_auc,
                    history, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':        epoch,
        'model_state':  model.state_dict(),
        'optim_state':  optimizer.state_dict(),
        'sched_state':  scheduler.state_dict(),
        'best_auc':     best_auc,
        'history':      history,
        'classes':      CLASSES,
        'num_classes':  NUM_CLASSES,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler and 'sched_state' in ckpt:
        scheduler.load_state_dict(ckpt['sched_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_auc', 0.0), \
           ckpt.get('history', {})


# ─── Main training function ───────────────────────────────────────────────────

def train(resume: bool = False):
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else
                          'cpu')
    print(f"Device: {device}")

    # ── Data ──
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    class_weights = class_weights.to(device)

    # ── Model ──
    model     = MultiModalFusionModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=LR_PATIENCE, verbose=True)
    writer    = SummaryWriter(log_dir=LOG_DIR)

    start_epoch = 0
    best_auc    = 0.0
    patience_counter = 0
    history = {k: [] for k in [
        'train_loss', 'train_acc', 'train_f1', 'train_auc',
        'val_loss',   'val_acc',   'val_f1',   'val_auc'
    ]}

    if resume and os.path.exists(LAST_MODEL_PATH):
        start_epoch, best_auc, history = load_checkpoint(
            LAST_MODEL_PATH, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    print(f"Model params: {model.count_parameters():,}")
    print(f"Training for up to {NUM_EPOCHS} epochs ...\n")

    # ── Epoch loop ──
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1, tr_auc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        vl_loss, vl_acc, vl_f1, vl_auc, vl_labels, vl_preds = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        scheduler.step(vl_auc)

        # Log
        for k, v in zip(
            ['train_loss','train_acc','train_f1','train_auc',
             'val_loss',  'val_acc',  'val_f1',  'val_auc'],
            [tr_loss, tr_acc, tr_f1, tr_auc,
             vl_loss, vl_acc, vl_f1, vl_auc]
        ):
            history[k].append(v)

        writer.add_scalars('Loss', {'train': tr_loss, 'val': vl_loss}, epoch)
        writer.add_scalars('AUC',  {'train': tr_auc,  'val': vl_auc},  epoch)
        writer.add_scalars('F1',   {'train': tr_f1,   'val': vl_f1},   epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] {elapsed:.0f}s | "
              f"Train — loss:{tr_loss:.4f} acc:{tr_acc:.4f} auc:{tr_auc:.4f} | "
              f"Val   — loss:{vl_loss:.4f} acc:{vl_acc:.4f} auc:{vl_auc:.4f}")

        # ── Checkpoint: last ──
        save_checkpoint(model, optimizer, scheduler, epoch + 1,
                        best_auc, history, LAST_MODEL_PATH)

        # ── Checkpoint: best ──
        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            best_auc, history, BEST_MODEL_PATH)
            print(f"  ★ New best AUC: {best_auc:.4f} — checkpoint saved")
            save_confusion_matrix(
                vl_labels, vl_preds,
                os.path.join(RESULTS_DIR, 'val_confusion_matrix.png')
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {EARLY_STOP} epochs)")
                break

    # ── Final test evaluation ──
    print("\n── Loading best checkpoint for test evaluation ──")
    load_checkpoint(BEST_MODEL_PATH, model)
    ts_loss, ts_acc, ts_f1, ts_auc, ts_labels, ts_preds = run_epoch(
        model, test_loader, criterion, optimizer, device, is_train=False
    )
    print(f"\nTest  — loss:{ts_loss:.4f} acc:{ts_acc:.4f} "
          f"f1:{ts_f1:.4f} auc:{ts_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(ts_labels, ts_preds,
                                 target_names=CLASSES, zero_division=0))

    save_confusion_matrix(
        ts_labels, ts_preds,
        os.path.join(RESULTS_DIR, 'test_confusion_matrix.png')
    )
    save_training_curves(
        history,
        os.path.join(RESULTS_DIR, 'training_curves.png')
    )

    # Save numeric results
    results = {
        'test_loss': ts_loss, 'test_acc': ts_acc,
        'test_f1':   ts_f1,   'test_auc': ts_auc,
        'best_val_auc': best_auc
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\nAll results saved to {RESULTS_DIR}/")
    print(f"Best model: {BEST_MODEL_PATH}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()
    train(resume=args.resume)
