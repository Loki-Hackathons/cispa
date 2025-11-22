#!/usr/bin/env python3
"""
Submission script for Task 2: Image Attribution
Submits submission.csv to the leaderboard server
"""

import requests
import os
from pathlib import Path

# Configuration
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
SERVER_URL = "http://34.122.51.94:80"
TASK_ID = "05-iar-attribution"
SUBMISSION_FILE = "submission.csv"

def submit():
    """Submit submission.csv to the leaderboard server"""
    
    if not os.path.exists(SUBMISSION_FILE):
        print(f"Error: {SUBMISSION_FILE} not found!")
        print("Please run training first to generate the submission file.")
        return False
    
    # Check file size
    file_size = os.path.getsize(SUBMISSION_FILE)
    print(f"Submission file: {SUBMISSION_FILE}")
    print(f"File size: {file_size / 1024:.2f} KB")
    
    if file_size > 10 * 1024 * 1024:  # 10 MB limit
        print("WARNING: File size exceeds 10 MB limit!")
        return False
    
    print(f"\nSubmitting to {SERVER_URL}/submit/{TASK_ID}...")
    
    try:
        with open(SUBMISSION_FILE, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/submit/{TASK_ID}",
                files={"file": f},
                headers={"X-API-Key": API_KEY},
                timeout=(10, 120)  # 10s connect, 120s read
            )
        
        print(f"Status code: {response.status_code}")
        
        try:
            result = response.json()
            print("Server response:", result)
            
            if response.status_code == 200:
                submission_id = result.get("submission_id")
                if submission_id:
                    print(f"\n✓ Submission successful! Submission ID: {submission_id}")
                else:
                    print("\n✓ Submission successful!")
                return True
            else:
                print("\n✗ Submission failed!")
                return False
                
        except Exception as e:
            print(f"Server response (text): {response.text}")
            print(f"Error parsing JSON: {e}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error submitting: {e}")
        return False

if __name__ == "__main__":
    submit()
