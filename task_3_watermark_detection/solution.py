import csv
import requests
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import torch.nn as nn
import torch.nn.functional as F
import random
import io
from torch.cuda.amp import autocast, GradScaler
import cv2

# ----------------------------
# CONFIG
# ----------------------------
DATASET_DIR = Path("Dataset")
SYNTHETIC_DIR = Path("Dataset_Synthetic")
USE_SYNTHETIC = True
SUBMISSION_FILE = "submission.csv"
LABELS = ["clean", "watermark"]
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
MODEL_CHECKPOINT = 'best_model_watermark_v5.pth'
IMG_SIZE = 512
CROP_SIZE = 448
NUM_WORKERS = 4

# Leaderboard submission
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
TASK_ID = "08-watermark-detection"

# ----------------------------
# ADVANCED WATERMARK GENERATION (Based on PDF techniques)
# ----------------------------
class WatermarkGenerator:
    """Génère des watermarks synthétiques avec techniques avancées du PDF"""
    
    def __init__(self):
        self.watermark_texts = [
            "© COPYRIGHT", "SAMPLE", "PREVIEW", "WATERMARK",
            "DO NOT COPY", "© 2024", "DRAFT", "CONFIDENTIAL",
            "PROPRIETARY", "LICENSED", "PROTECTED", "STOCK PHOTO",
            "FOR PREVIEW ONLY", "NOT FOR REPRODUCTION"
        ]
        self.positions = ['center', 'top-left', 'top-right', 'bottom-left', 
                         'bottom-right', 'diagonal', 'tiled', 'random']
        
    def add_dct_watermark(self, img: Image.Image, strength=10):
        """Watermark dans le domaine DCT (technique du PDF - section 3.2)"""
        img_array = np.array(img).astype(np.float32)
        
        # Appliquer DCT sur chaque canal
        watermarked = np.zeros_like(img_array)
        for channel in range(3):
            # DCT 2D par blocs de 8x8 (comme JPEG)
            h, w = img_array.shape[:2]
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = img_array[i:i+8, j:j+8, channel]
                    dct_block = cv2.dct(block)
                    
                    # Insérer watermark dans les coefficients moyens-hauts
                    # (éviter DC et très hautes fréquences)
                    watermark_pattern = np.random.randn(8, 8) * strength
                    # Masque pour coefficients moyens
                    mask = np.zeros((8, 8))
                    mask[2:6, 2:6] = 1
                    dct_block += watermark_pattern * mask
                    
                    watermarked[i:i+8, j:j+8, channel] = cv2.idct(dct_block)
        
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
        return Image.fromarray(watermarked)
    
    def add_dwt_watermark(self, img: Image.Image, strength=0.1):
        """Watermark dans le domaine DWT (technique du PDF - section 3.3)"""
        try:
            import pywt
            img_array = np.array(img).astype(np.float32)
            watermarked = np.zeros_like(img_array)
            
            for channel in range(3):
                # Décomposition en ondelettes (2 niveaux)
                coeffs = pywt.wavedec2(img_array[:,:,channel], 'haar', level=2)
                
                # Modifier les coefficients de détail du niveau 2 (HL, LH, HH)
                cA, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
                
                # Insérer watermark dans les détails de niveau 2
                watermark_h = np.random.randn(*cH2.shape) * strength * np.mean(np.abs(cH2))
                watermark_v = np.random.randn(*cV2.shape) * strength * np.mean(np.abs(cV2))
                
                cH2 += watermark_h
                cV2 += watermark_v
                
                # Reconstruction
                coeffs_modified = [cA, (cH2, cV2, cD2), (cH1, cV1, cD1)]
                watermarked[:,:,channel] = pywt.waverec2(coeffs_modified, 'haar')
            
            watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
            return Image.fromarray(watermarked)
        except ImportError:
            # Fallback si pywt n'est pas installé
            return self.add_dct_watermark(img, strength=10)
    
    def add_lsb_watermark(self, img: Image.Image):
        """LSB watermarking (technique du PDF - section 3.1)"""
        img_array = np.array(img).astype(np.uint8)
        h, w, c = img_array.shape
        
        # Générer un message binaire aléatoire
        message_length = (h * w) // 10  # 10% des pixels
        message = np.random.randint(0, 2, message_length)
        
        # Insérer dans les LSB du canal bleu (moins visible)
        flat_blue = img_array[:,:,2].flatten()
        for i, bit in enumerate(message):
            if i < len(flat_blue):
                flat_blue[i] = (flat_blue[i] & 0xFE) | bit
        
        img_array[:,:,2] = flat_blue.reshape(h, w)
        return Image.fromarray(img_array)
    
    def add_text_watermark(self, img: Image.Image, opacity_range=(50, 180)):
        """Watermark texte visible amélioré"""
        img_copy = img.copy().convert("RGBA")
        overlay = Image.new("RGBA", img_copy.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        text = random.choice(self.watermark_texts)
        font_size = int(img.size[1] * random.uniform(0.03, 0.10))
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        position = random.choice(self.positions)
        
        # Position tiled (répétition en grille)
        if position == 'tiled':
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            spacing_x = text_width + 100
            spacing_y = text_height + 100
            
            for x in range(-spacing_x, img.size[0] + spacing_x, spacing_x):
                for y in range(-spacing_y, img.size[1] + spacing_y, spacing_y):
                    opacity = random.randint(*opacity_range)
                    color = (255, 255, 255, opacity) if random.random() > 0.5 else (0, 0, 0, opacity)
                    draw.text((x, y), text, fill=color, font=font)
            
            composited = Image.alpha_composite(img_copy, overlay)
            return composited.convert("RGB")
        
        # Position diagonale améliorée
        elif position == 'diagonal':
            # Rotation du texte
            text_img = Image.new("RGBA", (img.size[0]*2, img.size[1]*2), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            for i in range(-img.size[0], img.size[0]*2, 200):
                for j in range(-img.size[1], img.size[1]*2, 200):
                    opacity = random.randint(*opacity_range)
                    text_draw.text((i, j), text, fill=(255, 255, 255, opacity), font=font)
            
            # Rotation de 45 degrés
            rotated = text_img.rotate(45, expand=False)
            # Crop au centre
            crop_x = (rotated.size[0] - img.size[0]) // 2
            crop_y = (rotated.size[1] - img.size[1]) // 2
            rotated = rotated.crop((crop_x, crop_y, crop_x + img.size[0], crop_y + img.size[1]))
            
            composited = Image.alpha_composite(img_copy, rotated)
            return composited.convert("RGB")
        
        # Positions standard
        else:
            # ...existing code for other positions...
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if position == 'center':
                x = (img.size[0] - text_width) // 2
                y = (img.size[1] - text_height) // 2
            elif position == 'random':
                x = random.randint(0, max(0, img.size[0] - text_width))
                y = random.randint(0, max(0, img.size[1] - text_height))
            else:
                margin = 20
                if 'top' in position:
                    y = margin
                else:
                    y = img.size[1] - text_height - margin
                if 'left' in position:
                    x = margin
                else:
                    x = img.size[0] - text_width - margin
            
            opacity = random.randint(*opacity_range)
            color = random.choice([(255, 255, 255, opacity), (0, 0, 0, opacity), 
                                  (200, 200, 200, opacity), (128, 128, 128, opacity)])
            draw.text((x, y), text, fill=color, font=font)
            
            composited = Image.alpha_composite(img_copy, overlay)
            return composited.convert("RGB")
    
    def add_logo_pattern_watermark(self, img: Image.Image):
        """Watermark pattern/logo amélioré avec formes variées"""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        pattern_type = random.choice(['circles', 'rectangles', 'grid', 'stars'])
        num_elements = random.randint(5, 15)
        
        for _ in range(num_elements):
            x = random.randint(0, img.size[0])
            y = random.randint(0, img.size[1])
            size = random.randint(20, 80)
            opacity = random.randint(30, 120)
            color = (random.randint(150, 255), random.randint(150, 255), 
                    random.randint(150, 255), opacity)
            
            if pattern_type == 'circles':
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           outline=color, width=2)
            elif pattern_type == 'rectangles':
                draw.rectangle([x-size, y-size, x+size, y+size], 
                             outline=color, width=2)
            elif pattern_type == 'grid':
                # Grille de lignes
                draw.line([x, y-size, x, y+size], fill=color, width=2)
                draw.line([x-size, y, x+size, y], fill=color, width=2)
        
        composited = Image.alpha_composite(img.convert("RGBA"), overlay)
        return composited.convert("RGB")
    
    def generate_watermark(self, img: Image.Image):
        """Génère un watermark avec distribution réaliste des techniques"""
        # Distribution basée sur les techniques du PDF
        technique = random.choices(
            ['text_visible', 'text_subtle', 'dct', 'dwt', 'lsb', 'logo'],
            weights=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
        )[0]
        
        if technique == 'text_visible':
            return self.add_text_watermark(img, opacity_range=(100, 200))
        elif technique == 'text_subtle':
            return self.add_text_watermark(img, opacity_range=(30, 80))
        elif technique == 'dct':
            return self.add_dct_watermark(img, strength=random.uniform(5, 15))
        elif technique == 'dwt':
            return self.add_dwt_watermark(img, strength=random.uniform(0.05, 0.2))
        elif technique == 'lsb':
            return self.add_lsb_watermark(img)
        else:
            return self.add_logo_pattern_watermark(img)

def create_synthetic_dataset(source_dir: Path, output_dir: Path, multiplier=3):
    """Crée un dataset synthétique avec watermarks"""
    print("Generating synthetic watermarked images...")
    generator = WatermarkGenerator()
    
    clean_dir = source_dir / "train" / "clean"
    watermark_dir = source_dir / "train" / "watermark"
    
    # Créer structure de sortie
    (output_dir / "train" / "clean").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "watermark").mkdir(parents=True, exist_ok=True)
    
    # Copier les images clean originales
    print("Copying clean images...")
    for img_path in clean_dir.glob("*"):
        if img_path.is_file():
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(output_dir / "train" / "clean" / img_path.name)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
    
    # Copier les watermark originaux
    print("Copying original watermarks...")
    for img_path in watermark_dir.glob("*"):
        if img_path.is_file():
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(output_dir / "train" / "watermark" / f"orig_{img_path.name}")
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
    
    # Générer watermarks synthétiques à partir des images clean
    print(f"Generating {multiplier} synthetic watermarks per clean image...")
    synthetic_count = 0
    for img_path in clean_dir.glob("*"):
        if not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            for i in range(multiplier):
                watermarked = generator.generate_watermark(img)
                output_path = output_dir / "train" / "watermark" / f"syn_{img_path.stem}_{i}.jpg"
                watermarked.save(output_path, quality=95)
                synthetic_count += 1
        except Exception as e:
            print(f"Error generating watermark for {img_path}: {e}")
    
    print(f"✓ Generated {synthetic_count} synthetic watermarks")
    
    # Copier validation tel quel
    print("Copying validation set...")
    for split in ["val"]:
        for cls in ["clean", "watermark"]:
            src_cls_dir = source_dir / split / cls
            dst_cls_dir = output_dir / split / cls
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            
            if src_cls_dir.exists():
                for img_path in src_cls_dir.glob("*"):
                    if img_path.is_file():
                        try:
                            img = Image.open(img_path).convert("RGB")
                            img.save(dst_cls_dir / img_path.name)
                        except:
                            pass

# ----------------------------
# ADVANCED AUGMENTATIONS
# ----------------------------
class WatermarkAugmentation:
    """Augmentations spécifiques basées sur les attaques du PDF"""
    
    @staticmethod
    def jpeg_compression(img, quality_range=(60, 98)):
        """Compression JPEG - attaque classique contre watermarks"""
        quality = random.randint(*quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    @staticmethod
    def add_gaussian_noise(img, sigma_range=(1, 15)):
        """Bruit gaussien - attaque contre watermarks"""
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def median_filter(img, kernel_size=None):
        """Filtre médian - attaque contre watermarks (PDF section 4.2)"""
        if kernel_size is None:
            kernel_size = random.choice([3, 5])
        arr = np.array(img)
        filtered = cv2.medianBlur(arr, kernel_size)
        return Image.fromarray(filtered)
    
    @staticmethod
    def gaussian_blur(img, sigma_range=(0.3, 2.5)):
        """Flou gaussien - attaque contre watermarks"""
        sigma = random.uniform(*sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    @staticmethod
    def scaling_attack(img, scale_range=(0.7, 1.3)):
        """Attaque par redimensionnement (PDF section 4.3)"""
        scale = random.uniform(*scale_range)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        scaled = img.resize(new_size, Image.LANCZOS)
        # Revenir à la taille originale
        return scaled.resize(img.size, Image.LANCZOS)
    
    @staticmethod
    def rotation_attack(img, angle_range=(-5, 5)):
        """Rotation légère (PDF section 4.3)"""
        angle = random.uniform(*angle_range)
        return img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    
    @staticmethod
    def brightness_contrast(img):
        """Ajustement luminosité/contraste"""
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        return img
    
    @staticmethod
    def sharpening(img):
        """Augmentation de netteté"""
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(random.uniform(0.5, 2.0))
    
    @staticmethod
    def histogram_equalization(img):
        """Égalisation d'histogramme - peut affecter watermarks"""
        arr = np.array(img)
        for i in range(3):
            arr[:,:,i] = cv2.equalizeHist(arr[:,:,i])
        return Image.fromarray(arr)

# ----------------------------
# FOCAL LOSS
# ----------------------------
class FocalLoss(nn.Module):
    """Focal Loss pour déséquilibre de classes"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        at = (1 - pt)**self.gamma
        fc_loss = (self.alpha * at * ce_loss).mean()
        return fc_loss

# ----------------------------
# MODEL, TRAINING, AND PREDICTION
# ----------------------------

def get_model(pretrained=True):
    """Retourne un modèle EfficientNet-B5 pré-entraîné et modifié"""
    model = models.efficientnet_b5(pretrained=pretrained)
    
    # Remplacer la dernière couche
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(LABELS))
    return model

def train_model():
    """Fonction principale pour l'entraînement du modèle"""
    
    # Création du dataset synthétique si activé
    if USE_SYNTHETIC:
        if not SYNTHETIC_DIR.exists() or not any((SYNTHETIC_DIR / "train" / "watermark").iterdir()):
            create_synthetic_dataset(DATASET_DIR, SYNTHETIC_DIR)
        train_dir = SYNTHETIC_DIR / "train"
        val_dir = SYNTHETIC_DIR / "val"
    else:
        train_dir = DATASET_DIR / "train"
        val_dir = DATASET_DIR / "val"

    # Transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([WatermarkAugmentation.jpeg_compression], p=0.3),
        transforms.RandomApply([WatermarkAugmentation.gaussian_blur], p=0.2),
        transforms.RandomApply([WatermarkAugmentation.add_gaussian_noise], p=0.2),
        transforms.RandomApply([WatermarkAugmentation.rotation_attack], p=0.2),
        transforms.RandomApply([WatermarkAugmentation.scaling_attack], p=0.2),
        WatermarkAugmentation.brightness_contrast,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets et DataLoaders
    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Modèle, critère et optimiseur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.to(device)
    
    # Data Parallelism pour multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc_train = correct_train / total_train

        # --- Validation ---
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_loss_val = running_loss_val / len(val_dataset)
        epoch_acc_val = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.4f} | Val Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}")

        # Sauvegarder le meilleur modèle
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            # Gérer DataParallel lors de la sauvegarde
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, MODEL_CHECKPOINT)
            print(f"✓ New best model saved with accuracy: {best_val_acc:.4f}")

def predict():
    """Génère les prédictions pour la soumission"""
    print("Starting prediction...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle
    model = get_model(pretrained=False)
    # Gérer DataParallel lors du chargement
    state_dict = torch.load(MODEL_CHECKPOINT)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Si le modèle a été sauvegardé avec DataParallel, les clés sont préfixées par 'module.'
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dir = DATASET_DIR / "test"
    test_files = sorted([p for p in test_dir.glob("*.png")])
    
    results = []

    for img_path in test_files:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = test_transforms(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                # Obtenir les scores softmax
                scores = torch.softmax(outputs, dim=1)
                # Score de la classe 'watermark' (index 1)
                watermark_score = scores[0][1].item()
            
            results.append([img_path.name, f"{watermark_score:.4f}"])
        except Exception as e:
            print(f"Error predicting {img_path.name}: {e}")
            results.append([img_path.name, "0.0000"]) # Fallback

    # Écrire le fichier de soumission
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'score'])
        writer.writerows(results)
    print(f"✓ Submission file created at {SUBMISSION_FILE}")


def submit():
    """Soumet le fichier de prédictions au serveur"""
    try:
        with open(SUBMISSION_FILE, 'rb') as f:
            files = {'file': (SUBMISSION_FILE, f)}
            data = {'task_id': TASK_ID, 'api_key': API_KEY}
            response = requests.post(SERVER_URL, files=files, data=data)
            
            if response.status_code == 200:
                print("✓ Successfully submitted! Results:")
                print(response.json())
            else:
                print(f"✗ Error submitting: {response.status_code}")
                print(response.text)
    except FileNotFoundError:
        print(f"✗ Submission file not found at {SUBMISSION_FILE}")
    except Exception as e:
        print(f"✗ An error occurred during submission: {e}")

if __name__ == "__main__":
    # --- Étape 1: Entraîner le modèle ---
    # Décommenter pour entraîner
    print("--- Starting Model Training ---")
    train_model()
    print("--- Model Training Finished ---")
    
    # --- Étape 2: Faire les prédictions ---
    print("\n--- Generating Predictions ---")
    predict()
    
    # --- Étape 3: Soumettre les résultats ---
    # Décommenter pour soumettre
    print("\n--- Submitting to Leaderboard ---")
    submit()