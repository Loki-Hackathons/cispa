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
NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

# Leaderboard submission
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"  # teams insert their assigned token here
TASK_ID = "08-watermark-detection"


def main():
    # ----------------------------
    # UNZIP DATASET
    # ----------------------------
    # if not DATASET_DIR.exists():
    #     print("Unzipping dataset...")
    #     with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    #         zip_ref.extractall(DATASET_DIR)
    # else:
    #     print("Dataset already extracted.")
    print("Assuming dataset is already extracted.")


    # ----------------------------
    # TRANSFORMS
    # ----------------------------
    # Augmentation de données pour l'entraînement
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),  # ← AJOUTÉ : manquait !
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Transform pour validation et test
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # ----------------------------
    # DATASETS & DATALOADERS
    # ----------------------------
    print("Loading datasets...")

    train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=DATASET_DIR / "val", transform=val_test_transform)

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

    test_dataset = TestDataset(DATASET_DIR / "test", transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")


    # ----------------------------
    # MODEL (ResNet50 - Plus performant)
    # ----------------------------
    print("Building model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_ftrs, len(LABELS))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # ----------------------------
    # TRAINING & EVALUATION
    # ----------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # Early stopping
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:
                print(f"  [Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        train_acc = 100 * train_correct / train_total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training Accuracy: {train_acc:.2f}%")

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ New best model saved! Accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Finished Training.")

    # Charger le meilleur modèle pour l'inférence
    print("Loading best model for inference...")
    model.load_state_dict(torch.load('best_model.pth'))

    # ----------------------------
    # INFERENCE
    # ----------------------------
    print("Generating predictions for submission...")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            image_names = batch["image_name"]
            
            outputs = model(images)
            # Convert outputs to probabilities (scores)
            scores = torch.softmax(outputs, dim=1)[:, 1] # Get score for 'watermark' class
            
            for fname, score in zip(image_names, scores):
                preds.append([fname, score.item()])

    # ----------------------------
    # SAVE SUBMISSION
    # ----------------------------
    print(f"Saving predictions to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "score"])  # not label
        writer.writerows(preds)

    print(f"Successfully saved submission file to {SUBMISSION_FILE}")
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

if __name__ == '__main__':
    main()