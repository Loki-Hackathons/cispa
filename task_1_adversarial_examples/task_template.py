import torch
import requests
import sys
import os
import numpy as np

# --------------------------------
# DATASET
# --------------------------------

"""
Dataset contents:

-"image_ids": Tensor containing the IDs of the 100 natural images, has shape (100)
-"images": Tensor containing the 100 natural images, has shape (100, 3, 28, 28)
-"labels": Tensor of true labels for the images, has shape (100)
"""

# Load the dataset
dataset = torch.load("natural_images.pt", weights_only=False)

print("Dataset keys:", dataset.keys())
print("Image IDs shape:", dataset["image_ids"].shape)
print("Images shape:", dataset["images"].shape)
print("Labels shape:", dataset["labels"].shape)
print("First 10 image IDs:", dataset["image_ids"][:10])
print("First 10 labels:", dataset["labels"][:10])
print("First image tensor:", dataset["images"][:1])

# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .npz file of the following format:

-"image_ids": Tensor containing the IDs of the images corresponding to your adversarial examples, has shape (100)
-"images": Tensor containing the generated adversarial examples in the same order as the corresponding
           natural images, has shape (100, 3, 28, 28)
"""

# Example Submission:

adversarial_examples = torch.randint(0, 256, size=(100, 3, 28, 28), dtype=torch.uint8)

adversarial_examples = adversarial_examples.float() / 255.0  # normalize to [0, 1] range

images_np = adversarial_examples.detach().cpu().numpy()

image_ids = np.arange(len(images_np))

np.savez_compressed("example_submission.npz", image_ids=image_ids, images=images_np)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

"""
Adversarial Examples Task — Participant Submission Guide
========================================================

You will upload a single **.npz** file that contains ONLY an array named **'images'**.
The evaluator will load your file, run shape/dtype checks against the natural images,
and then score it by running a fixed classifier and measuring perturbations.

Follow these rules carefully to avoid automatic rejection.

1) File format
--------------
- **Extension:** `.npz` (NumPy compressed archive)
- **Content:** must contain exactly one required key: `'images'`
- **Max file size:** 200 MB (hard limit). Larger files are rejected.

2) Array requirements
---------------------
Let `G` be the ground-truth tensor loaded:

- **Shape:** `images.shape` must match `G["images"].shape` **exactly**.
  - If `G["images"]` is `(N, 3, H, W)`, your array must also be `(N, 3, H, W)`.
  - No extra samples; no fewer; no different dimensions.
- **Dtype:** `images.dtype` must match `G["images"].dtype` **exactly**.
  - If the GT uses `float32`, you must submit `float32`.
  - Safe cast example: `images = np.asarray(images, dtype=np.float32)`
- **Finite values only:** No NaN or Inf anywhere.
  - The evaluator checks: `torch.isfinite(images).all()`.
- **Contiguity:** The server will convert to a contiguous Torch tensor; standard NumPy arrays are fine.


3) Typical failure messages & what they mean
--------------------------------------------
- "File must be .npz and contain an 'images' array."
  → Wrong extension or missing `'images'` key.
- "File too large: X bytes (limit 209715200)."
  → Your file exceeds 200 MB.
- "Failed to read .npz: ..."
  → The file is corrupted or not a valid `.npz` created with `allow_pickle=False`.
- "Failed to convert 'images' to torch tensor: ..."
  → Your `'images'` array has an unsupported dtype or structure (e.g., object array).
- "Submitted images must have shape (N, C, H, W), but got (...)."
  → Shape mismatch with the ground-truth images.
- "Submitted images must be of type torch.float32, but got torch.float64."
  → Dtype mismatch with the ground-truth images.
- "Images must not contain NaN or Inf values."
  → Clean your array: `np.isfinite(images).all()` must be True.
"""

BASE_URL  = "http://34.122.51.94:80"
API_KEY  = "f62b1499d4e2bf13ae56be5683c974c1"  

TASK_ID = "10-adversarial-examples"

# Path to the .npz file containing the images you want to get logits for

QUERY_PATH = "PATH/TO/YOUR/QUERY_FILE.npz"

# Path to the .npz file you want to send

FILE_PATH = "PATH/TO/YOUR/SUBMISSION.npz"

GET_LOGITS = False      # set True to get logits from the API
SUBMIT     = False      # set True to submit your solution

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if GET_LOGITS:
    with open(QUERY_PATH, "rb") as f:
        files = {"npz": (QUERY_PATH, f, "application/octet-stream")}
        response = requests.post(
            f"{BASE_URL}/{TASK_ID}/logits",
            files=files,
            headers={"X-API-Key": API_KEY},
        )

    if response.status_code == 200:
        data = response.json()
        print("Request successful")
        print(data)

    else:
        print("Request failed")
        print("Status code:", response.status_code)
        print("Detail:", response.text)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120), 
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)