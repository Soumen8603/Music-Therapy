from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .core import BEHAVIOR_LABELS, BehaviorClassifier, BehaviorVideoDataset


def train(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Behavior data root does not exist: {data_root}. "
            "This app prefers a pretrained autism behavior detector checkpoint at artifacts/behavior_model.pt. "
            "Use a custom dataset only if you want to train your own model."
        )

    dataset = BehaviorVideoDataset(
        data_root,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        train=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training videos found under {data_root}. "
            "Expected class folders like normal/ and fight/."
        )

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BehaviorClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - train"):
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.size(0))

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for clips, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - val"):
                clips = clips.to(device)
                labels = labels.to(device)
                logits = model(clips)
                predictions = logits.argmax(dim=1)
                val_correct += int((predictions == labels).sum().item())
                val_total += int(labels.size(0))

        train_accuracy = train_correct / max(1, train_total)
        val_accuracy = val_correct / max(1, val_total)
        if val_accuracy >= best_val:
            best_val = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": getattr(dataset, "class_names", BEHAVIOR_LABELS),
                    "clip_len": args.clip_len,
                    "frame_stride": args.frame_stride,
                },
                output_path,
            )
        print(
            f"epoch={epoch} train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} best_val={best_val:.4f}"
        )

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an optional custom behavior model. The web app is designed to use a pretrained checkpoint for autism behavior detection."
    )
    parser.add_argument("--data-root", required=True, help="Directory with behavior class subfolders.")
    parser.add_argument("--output", default="artifacts/behavior_model.pt", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
