import csv
import random
import zipfile
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image


# ----------------------------
# CONFIG
# ----------------------------
ZIP_FILE = "Dataset.zip"        # Path to dataset zip file
DATASET_DIR = Path("dataset")   # Folder after extraction
SUBMISSION_FILE = "submission.csv"
LABELS = ["clean", "watermark"]

# Leaderboard submission
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"  # teams insert their assigned token here
TASK_ID = "08-watermark-detection"


# ----------------------------
# UNZIP DATASET
# ----------------------------
if not DATASET_DIR.exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
else:
    print("Dataset already extracted.")


# ----------------------------
# TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])


# ----------------------------
# DATASETS & DATALOADERS
# ----------------------------
print("Loading datasets...")

train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=transform)
val_dataset   = datasets.ImageFolder(root=DATASET_DIR / "val", transform=transform)

# Custom dataset for unlabeled test images
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.*")))  # all image files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "image_name": img_path.name}

test_dataset = TestDataset(DATASET_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")


# ----------------------------
# EXAMPLE MODEL (ResNet18)
# ----------------------------
print("Building dummy model...")
model = models.resnet18(weights=None, num_classes=len(LABELS))  # untrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# ----------------------------
# DUMMY INFERENCE / RANDOM SCORES
# ----------------------------
print("Generating random prediction scores for submission...")
preds = []
for batch in test_loader:
    for fname in batch["image_name"]:
        score = round(random.random(), 4)  # random float in [0,1]
        preds.append([fname, score])

# ----------------------------
# SAVE SUBMISSION
# ----------------------------
with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "score"])  # not label
    writer.writerows(preds)

print(f"Saved submission file to {SUBMISSION_FILE}")
print("Format: image_name,score | Allowed scores: [0,1]")


# ----------------------------
# SUBMIT TO LEADERBOARD SERVER
# ----------------------------
if API_KEY is None:
    print("No TOKEN provided. Please set your team TOKEN in this script to submit.")
else:
    print("Submitting to leaderboard server...")

    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Server response:", response.json())
