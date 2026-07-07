# Emotion and Behavior Detection Model

This project now uses a local PyTorch pipeline for two tasks:

- Facial emotion detection from images/webcam snapshots
- Behavior detection from CCTV or uploaded video clips

## Recommended datasets

Emotion detection:

- RAF-DB, because it is a strong facial expression dataset with the 7 basic emotions and is well-suited for transfer learning.

Behavior detection:

- A pretrained autism-focused video behavior model is preferred for this app, rather than a generic violence dataset.

## Model design

- Emotion model: MobileNetV3-small image classifier
- Behavior model: MobileNetV3-small frame encoder plus temporal GRU over sampled video frames

## Expected folder structure

Emotion dataset:

```text
data/rafdb/
  angry/
  disgust/
  fear/
  happy/
  sad/
  surprise/
  neutral/
```

Behavior dataset:

```text
# Primary behavior detection should use a pretrained autism behavior model checkpoint.
# Custom dataset training is optional and not required for app usage.
```

## Training

First create the folder layout:

```powershell
python scripts/prepare_dataset_layout.py
```

Then copy the downloaded dataset contents into those folders.

If your emotion dataset is still in its extracted vendor layout, use the importer:

```powershell
python scripts/import_datasets.py --emotion-source C:\path\to\RAF-DB
```

If you have a custom behavior dataset and want to train a custom model, you can still use the importer to copy it into the app layout. The app itself is designed to work with a pretrained autism behavior model checkpoint instead of RWF-2000 by default.

The importer expects class-like subfolder names. If your download uses a different naming scheme, rename the folders first or point the importer at the class-level directories.

Train the emotion model:

```powershell
python -m emotion_behavior.train_emotion --data-root data/rafdb --output artifacts/emotion_model.pt
```

Train the behavior model (optional custom dataset):

```powershell
python -m emotion_behavior.train_behavior --data-root data/behavior --output artifacts/behavior_model.pt
```

> The app is designed to use a pretrained autism behavior detector checkpoint by default. If you do not have a custom behavior dataset, place the checkpoint at `artifacts/behavior_model.pt` or set `BEHAVIOR_MODEL_PATH` in `.env`.

## Runtime behavior

- If `artifacts/emotion_model.pt` exists, webcam snapshots and uploaded images will use the trained emotion model.
- If `artifacts/behavior_model.pt` exists, uploaded videos and CCTV/stream URLs will use the trained behavior model.
- If a checkpoint is missing, the app keeps running and reports that the analyzer is unavailable.
