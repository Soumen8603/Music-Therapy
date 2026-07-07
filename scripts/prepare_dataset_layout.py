from __future__ import annotations

from pathlib import Path


EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def make_dirs(base: Path, classes: list[str]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        (base / class_name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    root = Path("data")
    make_dirs(root / "rafdb", EMOTION_CLASSES)
    print("Created dataset layout under ./data")
    print("")
    print("Place RAF-DB images here:")
    for class_name in EMOTION_CLASSES:
        print(f"  data/rafdb/{class_name}/")
    print("")
    print("Behavior detection uses a pretrained checkpoint rather than the RWF-2000 dataset.")
    print("Place a pretrained behavior model at artifacts/behavior_model.pt or set BEHAVIOR_MODEL_PATH in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
