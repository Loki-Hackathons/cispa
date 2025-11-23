import sys
import subprocess

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "local"
        
    # Paths relative to task_1_adversarial_examples/
    script_path = "analyze_submission.py"
    submission_path = "version2/submission.npz"
    
    print(f"Running analysis on {submission_path} (Mode: {mode})")
    subprocess.run(["python", script_path, submission_path, "--mode", mode])

if __name__ == "__main__":
    main()
