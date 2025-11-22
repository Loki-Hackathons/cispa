import requests
import torch
import numpy as np
import json
import os
import sys
from datetime import datetime
import argparse

# Configuration
BASE_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
TASK_ID = "10-adversarial-examples"
LOG_DIR = "logs"
DATASET_PATH = "natural_images.pt"

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)
    return torch.load(DATASET_PATH, weights_only=False)

def get_logits_from_api(submission_path):
    print(f"Querying API for logits: {submission_path}")
    with open(submission_path, "rb") as f:
        files = {"npz": (submission_path, f, "application/octet-stream")}
        try:
            response = requests.post(
                f"{BASE_URL}/{TASK_ID}/logits",
                files=files,
                headers={"X-API-Key": API_KEY},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)
            sys.exit(1)

def calculate_score_api(logits_data, submission_path, dataset):
    # 1. Load Data
    true_labels = dataset["labels"].numpy()
    original_images = dataset["images"].numpy()
    
    # Load Submission
    sub = np.load(submission_path)
    adv_images = sub["images"]
    
    # 2. Parse Logits
    results = logits_data["results"]
    # Sort results by image_id just in case
    results.sort(key=lambda x: x["image_id"])
    
    scores_raw = []
    scores_norm = []
    success_count = 0
    # Normalization factor for L2 (so that distance is in [0,1])
    # Each pixel/channel is in [0,1], so max L2 per image is sqrt(C*H*W)
    _, C, H, W = original_images.shape
    l2_norm_factor = float(np.sqrt(C * H * W))
    
    print("\n--- Analysis ---")
    
    for i, res in enumerate(results):
        img_id = res["image_id"]
        logits = np.array(res["logits"])
        predicted_class = np.argmax(logits)
        true_class = true_labels[img_id]
        
        # Check Misclassification
        is_misclassified = (predicted_class != true_class)
        
        # Calculate L2 Distance
        # Ensure shapes match
        diff = adv_images[i] - original_images[i]
        l2_dist = np.linalg.norm(diff)
        l2_dist_norm = l2_dist / l2_norm_factor
        l2_dist_norm = float(np.clip(l2_dist_norm, 0.0, 1.0))
        
        # Score Rule
        # If misclassified: Score = L2 Distance (raw and normalized variants)
        # If correctly classified: Score = 1.0 (Penalty, normalized by definition)
        if is_misclassified:
            final_score_raw = l2_dist
            final_score_norm = l2_dist_norm
            success_count += 1
        else:
            final_score_raw = 1.0
            final_score_norm = 1.0
            
        scores_raw.append(final_score_raw)
        scores_norm.append(final_score_norm)
    
    avg_score_raw = float(np.mean(scores_raw))
    avg_score_norm = float(np.mean(scores_norm))
    success_rate = (success_count / 100) * 100
    
    print(f"Success Rate (Misclassified): {success_rate}%")
    print(f"Average L2 Distance (Raw, successes only):       {np.mean([s for s in scores_raw if s != 1.0] or [0.0]):.4f}")
    print(f"Average L2 Distance (Normalized, successes only): {np.mean([s for s in scores_norm if s != 1.0] or [0.0]):.4f}")
    print(f"ESTIMATED LEADERBOARD SCORE (normalized [0,1]):   {avg_score_norm:.4f} (Lower is better)")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "submission_file": submission_path,
        "leaderboard_score_normalized": avg_score_norm,
        "leaderboard_score_raw": avg_score_raw,
        "success_rate": success_rate,
        "mode": "api",
        "details": results
    }

def calculate_score_local(submission_path, dataset):
    """
    Local-only estimate without calling the API.
    - We cannot know misclassification w.r.t. the black-box model.
    - We report the lower-bound score = mean normalized L2 (assuming all succeed).
    """
    original_images = dataset["images"].numpy()
    sub = np.load(submission_path)
    adv_images = sub["images"]

    _, C, H, W = original_images.shape
    l2_norm_factor = float(np.sqrt(C * H * W))

    diffs = adv_images - original_images
    l2_per_image = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    l2_per_image_norm = np.clip(l2_per_image / l2_norm_factor, 0.0, 1.0)

    score_lower_bound = float(np.mean(l2_per_image_norm))

    print("\n--- Local Analysis (no API) ---")
    print(f"Average L2 Distance (Normalized): {score_lower_bound:.4f}")
    print("Note: This is a LOWER BOUND. Actual leaderboard score can only be worse (never better),")
    print("      because correctly classified samples are set to 1.0 by the evaluator.")

    return {
        "timestamp": datetime.now().isoformat(),
        "submission_file": submission_path,
        "leaderboard_score_lower_bound_normalized": score_lower_bound,
        "mode": "local",
        "details": []
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to submission .npz file", default="submission_pgd.npz", nargs="?")
    parser.add_argument("--mode", choices=["api", "local"], default="api", help="Use 'api' to query logits from server, or 'local' for local-only L2 estimate")
    args = parser.parse_args()
    
    submission_file = args.file
    if not os.path.exists(submission_file):
        print(f"File not found: {submission_file}")
        return

    # Load Dataset for Comparison
    dataset = load_dataset()
    
    if args.mode == "api":
        # 1. Get Logits from API
        api_response = get_logits_from_api(submission_file)
        # 2. Calculate Score (normalized)
        analysis = calculate_score_api(api_response, submission_file, dataset)
    else:
        # Local-only estimation (no API calls)
        analysis = calculate_score_local(submission_file, dataset)
    
    # 4. Save Log
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = args.mode
    log_path = os.path.join(LOG_DIR, f"analysis_{timestamp}_{suffix}.json")
    
    with open(log_path, "w") as f:
        json.dump(analysis, f, indent=2)
        
    print(f"\nFull analysis saved to: {log_path}")

if __name__ == "__main__":
    main()

