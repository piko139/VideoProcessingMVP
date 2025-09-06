# tictactoe/utils.py

import os
import sys
import random
import numpy as np
import torch

def select_device() -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    """Set global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def env_summary() -> str:
    device = select_device()
    lines = [
        "Tic-Tac-Toe NN — Environment Summary",
        f"Python: {sys.version.split()[0]}",
        f"NumPy: {np.__version__}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"Selected device: {device}",
    ]
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            lines += [
                f"GPU: {name} (CC {cap[0]}.{cap[1]})",
                f"GPU total memory: {mem_total:.2f} GB",
            ]
        except Exception:
            pass
    return "\n".join(lines)

# Constants
BOARD_SIZE = 3
N_ACTIONS = BOARD_SIZE * BOARD_SIZE  # 9
PLAYER_X, PLAYER_O, EMPTY = 1, -1, 0