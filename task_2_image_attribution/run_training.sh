#!/bin/bash
# Script to run training with proper environment setup

cd /p/home/jusers/dougnon1/jureca/code/cispa-hackathon/task_2_image_attribution

# Load necessary modules
module load GCC
module load CUDA
module load PyTorch
module load torchvision

# Run training
python3 main_attribution.py 2>&1 | tee training.log

# After training, submit if submission.csv exists
if [ -f "submission.csv" ]; then
    echo "Submitting results..."
    python3 -c "
import requests
import os

API_KEY = 'f62b1499d4e2bf13ae56be5683c974c1'
SERVER_URL = 'http://34.122.51.94:80'
TASK_ID = '05-iar-attribution'
SUBMISSION_FILE = 'submission.csv'

if os.path.exists(SUBMISSION_FILE):
    print(f'Submitting {SUBMISSION_FILE}...')
    with open(SUBMISSION_FILE, 'rb') as f:
        response = requests.post(
            f'{SERVER_URL}/submit/{TASK_ID}',
            files={'file': f},
            headers={'X-API-Key': API_KEY},
        )
    print(f'Status code: {response.status_code}')
    try:
        print('Server response:', response.json())
    except:
        print('Server response (text):', response.text)
else:
    print('Submission file not found!')
"
fi

