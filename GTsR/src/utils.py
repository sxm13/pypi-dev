from __future__ import annotations

import shutil
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state, is_best: bool, checkpoint_path: str | Path, best_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    best_path = Path(best_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


def metric_functions(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred = (torch.sigmoid(logits.detach()) >= threshold).cpu().numpy()
    truth = (target.detach() >= 0.5).cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "recall": recall_score(truth, pred, zero_division=0),
        "f1": f1_score(truth, pred, zero_division=0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    