"""
Дообучение новой головы PP-LCNet на объединённом синтетическом наборе
(старый data/train + расширенный data/train_v3).

Запуск:
    python train_head_v3.py --arch x1 --epochs 5 --workers 4

Файлы:
    weights/x1/                — предобученный backbone + процессор
    data/train/{0,180}_degree/ — старый набор
    data/train_v3/{0,180}_degree/ — расширенный набор
    outputs/best_head_v3_x1.pth — чекпоинт
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CombinedSynthDataset(Dataset):
    """Читает пары (path, label) из нескольких корневых папок.
    Каждая корневая папка должна содержать подпапки 0_degree/ и 180_degree/.
    """

    def __init__(self, roots, processor):
        self.samples = []
        for root in roots:
            root = Path(root)
            for label, subdir in [(0, "0_degree"), (1, "180_degree")]:
                for p in (root / subdir).glob("*.png"):
                    self.samples.append((p, label))
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)


def build_model(weights_dir: Path, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(
        str(weights_dir), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(weights_dir),
        num_labels=2,
        ignore_mismatched_sizes=True,   # новая голова с нуля
        local_files_only=True,
    )
    model.to(device)
    return model, processor


def build_optimizer(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if "classifier" in name or "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(pixel_values=x).last_hidden_state
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu()
        preds.append(probs)
        targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets).float()
    return ((preds - targets) ** 2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["x0_25", "x1"], default="x1")
    ap.add_argument("--data-roots", nargs="+",
                    default=["data/train", "data/v3"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_dir = Path("weights/x1") if args.arch == "x1" else Path("weights")
    out_path = Path(args.out) if args.out else Path(f"outputs/best_head_v3_{args.arch}.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = build_model(weights_dir, device)
    optimizer = build_optimizer(
        model, args.lr_backbone, args.lr_head, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    full_ds = CombinedSynthDataset(args.data_roots, processor)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Всего: {len(full_ds)} | Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    best_brier = float("inf")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(pixel_values=x).last_hidden_state
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        brier = evaluate(model, val_loader, device)
        score = 1 - brier
        print(f"Epoch {epoch+1}: val 1-Brier = {score:.4f}")

        if brier < best_brier:
            best_brier = brier
            torch.save(model.state_dict(), out_path)
            print(f"  → сохранено: {out_path}")

    print(f"\nBest val 1-Brier: {1 - best_brier:.4f}")
    print(f"Веса: {out_path}")


if __name__ == "__main__":
    main()