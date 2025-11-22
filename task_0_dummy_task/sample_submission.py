import csv
import random
import requests
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
NUM_ROWS = 200
SUBMISSION_FILE = "submission.csv"

# Leaderboard submission
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"  # teams insert their assigned token here
TASK_ID = "00-dummy-task"


# ----------------------------
# DUMMY INFERENCE / RANDOM SCORES
# ----------------------------
print("Generating random prediction scores for submission...")
preds = []
for i in range(NUM_ROWS):
    score = round(random.random(), 4)  # random float in [0,1]
    preds.append([i, score])


# ----------------------------
# SAVE SUBMISSION
# ----------------------------
with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    writer.writerows(preds)

print(f"Saved submission file to {SUBMISSION_FILE}")
print("Format: id,label | Allowed labels: [0,1] (probability)")


# ----------------------------
# SUBMIT TO LEADERBOARD SERVER
# ----------------------------
if API_KEY is None:
    print("No API_KEY provided. Please set your team API_KEY in this script to submit.")
else:
    print("Submitting to leaderboard server...")

    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Server response:", response.json())
