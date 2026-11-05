"""Run a command while holding the shared GPU lock (exactmoe.runtime.gpu_lock)."""
import subprocess, sys
sys.path.insert(0, r"C:\Users\yuval\Documents\GitHub\ExactMoE")
from exactmoe.runtime.gpu_lock import GpuLock
with GpuLock(holder="claude-code session (HOMELAB/ExactMoE)", purpose=" ".join(sys.argv[1:])[:200]):
    sys.exit(subprocess.call(sys.argv[1:]))
