import requests
import os
import sys
import argparse

BASE_URL  = "http://34.122.51.94:80"
API_KEY  = "f62b1499d4e2bf13ae56be5683c974c1"  
TASK_ID = "10-adversarial-examples"
# Path relative to task_1_adversarial_examples/
FILE_PATH = "version2/submission.npz"

def submit():
    if not os.path.isfile(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)

    print(f"Submitting {FILE_PATH} to {TASK_ID}...")
    
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
            
        if resp.status_code == 200:
            print("Success!")
            print(resp.json())
        else:
            print(f"Error {resp.status_code}: {resp.text}")
            
    except Exception as e:
        print(f"Submission failed: {e}")

if __name__ == "__main__":
    submit()
