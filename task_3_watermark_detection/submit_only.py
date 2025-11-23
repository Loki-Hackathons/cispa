#!/usr/bin/env python3
"""
Script pour soumettre le fichier submission.csv à la tâche 3
"""
import requests
from pathlib import Path

# Configuration
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
TASK_ID = "08-watermark-detection"

# Chemins possibles pour le fichier de soumission
submission_paths = [
    "/p/scratch/training2557/dougnon1/submission_task3.csv",
    "submission.csv",
    "/p/home/jusers/dougnon1/jureca/code/cispa-hackathon/task_3_watermark_detection/submission.csv"
]

def submit():
    # Trouver le fichier de soumission
    submission_file = None
    for path in submission_paths:
        if Path(path).exists():
            submission_file = path
            print(f"✓ Fichier trouvé: {submission_file}")
            break
    
    if submission_file is None:
        print("❌ Erreur: Aucun fichier submission.csv trouvé!")
        print("Chemins vérifiés:")
        for path in submission_paths:
            print(f"  - {path}")
        return
    
    # Vérifier le contenu
    with open(submission_file, 'r') as f:
        lines = f.readlines()
        print(f"✓ Fichier contient {len(lines)} lignes (en-tête + {len(lines)-1} prédictions)")
        if len(lines) > 1:
            print(f"  Première ligne: {lines[0].strip()}")
            print(f"  Exemple: {lines[1].strip()}")
    
    # Soumettre
    print(f"\n📤 Soumission du fichier à {SERVER_URL}/submit/{TASK_ID}...")
    try:
        with open(submission_file, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/submit/{TASK_ID}",
                files={"file": f},
                headers={"X-API-Key": API_KEY},
            )
        
        print(f"✓ Réponse du serveur: {response.status_code}")
        result = response.json()
        print(f"📊 Résultat: {result}")
        
        if "score" in result:
            print(f"\n🎉 Score obtenu: {result.get('score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la soumission: {e}")

if __name__ == "__main__":
    submit()


