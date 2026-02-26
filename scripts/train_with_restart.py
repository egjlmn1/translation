"""Auto-restart wrapper for training. Catches CUDA crashes and resumes from latest checkpoint."""
import subprocess
import sys
import time
import os

MAX_RETRIES = 100  # Maximum number of auto-restarts
WAIT_SECONDS = 5   # Wait between restarts to let GPU cool/reset

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_latest.pt")
    checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_latest.pt")
    
    # Forward any extra args (like --with-server)
    extra_args = sys.argv[1:]
    
    for attempt in range(1, MAX_RETRIES + 1):
        # Build command
        cmd = [sys.executable, os.path.join(project_root, "scripts", "train_model.py")]
        
        # Always resume from checkpoint if it exists (after first crash)
        if os.path.exists(checkpoint_path):
            cmd += ["--resume", checkpoint_path]
        
        cmd += extra_args
        
        print(f"\n{'='*60}")
        print(f"[AutoRestart] Attempt {attempt}/{MAX_RETRIES}")
        print(f"[AutoRestart] Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print(f"\n[AutoRestart] Training completed successfully!")
            return 0
        
        # Check if it was a CUDA error (returncode != 0)
        print(f"\n[AutoRestart] Process exited with code {result.returncode}")
        print(f"[AutoRestart] Waiting {WAIT_SECONDS}s before restart...")
        time.sleep(WAIT_SECONDS)
    
    print(f"\n[AutoRestart] Exceeded {MAX_RETRIES} retries. Giving up.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
