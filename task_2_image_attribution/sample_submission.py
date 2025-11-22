"""
Example submission script for the IAR Attribution Task.

Submission Requirements (read carefully to avoid automatic rejection):

1. CSV FORMAT
----------------
- The file **must be a CSV** with extension `.csv`.
- It must contain **exactly two columns**, named:
      image_name, label
  → Column names must match exactly (lowercase, no extra spaces).
  → Column order does not matter, but both must be present.
  → Both columns must be string-typed (pandas `string` or `object` is fine).

2. ROW COUNT AND FILENAMES
-----------------------------
- Your file must contain **exactly 8,000 rows**.
- Each row corresponds to one unique `image_name` present in the official ground truth.
- Every `image_name` from the ground truth must appear **exactly once** in your CSV.
- Do **not** add, remove, or rename any `image_name` values.
- Do **not** include duplicates or missing entries.

3. LABEL VALUES (CASE-SENSITIVE)
---------------------------------
- Each `label` must be **exactly one** of the following allowed classes (case-sensitive):

  [
    "outlier",
    "VAR",
    "RAR",
    "Taming"
  ]

- Do not use aliases, capitalization changes, or extra spaces (e.g., "Outlier", "var", "RAR ", "taming" will be rejected).
- Each prediction must be exactly one of the above labels.

4. TECHNICAL LIMITS
----------------------
- Maximum file size: **10 MB**
- Encoding: **UTF-8** recommended.
- Avoid formulas, extra columns, hidden columns, BOM headers, or blank lines.
- Ensure both columns are strings (if needed: `df = df.astype({"image_name": "string", "label": "string"})`).

5. VALIDATION SUMMARY
------------------------
Your submission will fail if:
- Columns don’t match exactly `["image_name", "label"]`
- Row count differs from **8,000**
- Any `image_name` is missing or duplicated
- Any unexpected `image_name` appears (i.e., not in ground truth)
- Any `label` is outside the allowed set `{"outlier","VAR","RAR","Taming"}`
- File is too large or not a CSV
- CSV cannot be parsed
"""


import os
import sys
import requests

BASE_URL  = "http://34.122.51.94:9000"
API_KEY   = "f62b1499d4e2bf13ae56be5683c974c1"  # replace with your actual API key

TASK_ID   = "05-iar-attribution"
FILE_PATH = "Your-Submission-File.csv"  # replace with your actual file path

SUBMIT     = False

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                # (fieldname) -> (filename, fileobj, content_type)
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),  # (connect timeout, read timeout)
            )
        # Helpful output even on non-2xx
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