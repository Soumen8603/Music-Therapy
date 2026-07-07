from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EMOTION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BEHAVIOR_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

EMOTION_TARGETS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
BEHAVIOR_TARGETS = ["normal", "fight"]

EMOTION_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}

BEHAVIOR_ALIASES = {
    "fight": "fight",
    "violence": "fight",
    "violent": "fight",
    "nonfight": "normal",
    "non_fight": "normal",
    "normal": "normal",
    "nonviolent": "normal",
    "non_violent": "normal",
}


def _copy_tree(source: Path, target: Path, allowed_extensions: set[str]) -> int:
    copied = 0
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            destination = target / path.name
            if destination.exists():
                stem = destination.stem
                suffix = destination.suffix
                index = 1
                while destination.exists():
                    destination = target / f"{stem}_{index}{suffix}"
                    index += 1
            shutil.copy2(path, destination)
            copied += 1
    return copied


def import_emotion_dataset(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for source_dir in source_root.iterdir():
        if not source_dir.is_dir():
            continue
        class_name = EMOTION_ALIASES.get(source_dir.name.lower().replace(" ", ""))
        if not class_name:
            continue
        copied = _copy_tree(source_dir, target_root / class_name, EMOTION_EXTENSIONS)
        total += copied
        print(f"emotion {source_dir.name} -> {class_name}: {copied}")
    print(f"Emotion files copied: {total}")


def import_behavior_dataset(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for source_dir in source_root.iterdir():
        if not source_dir.is_dir():
            continue
        class_name = BEHAVIOR_ALIASES.get(source_dir.name.lower().replace(" ", "").replace("-", "").replace("_", ""))
        if not class_name:
            continue
        copied = _copy_tree(source_dir, target_root / class_name, BEHAVIOR_EXTENSIONS)
        total += copied
        print(f"behavior {source_dir.name} -> {class_name}: {copied}")
    print(f"Behavior files copied: {total}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import extracted datasets into the app layout.")
    parser.add_argument("--emotion-source", type=Path, help="Path to extracted RAF-DB emotion folders.")
    parser.add_argument(
        "--behavior-source",
        type=Path,
        help="Optional path to a custom behavior dataset if you want to train your own model."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Target data root.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.emotion_source:
        import_emotion_dataset(args.emotion_source, args.data_root / "rafdb")
    if args.behavior_source:
        import_behavior_dataset(args.behavior_source, args.data_root / "behavior")
    if not args.emotion_source and not args.behavior_source:
        print("Provide --emotion-source and/or --behavior-source")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
