# tictactoe/training.py

import numpy as np
import matplotlib.pyplot as plt
from .utils import BOARD_SIZE

def _rolling_mean(arr, window: int = 100):
    """Simple centered rolling mean; falls back to valid if window > len(arr)."""
    a = np.asarray(arr, dtype=np.float32)
    if len(a) == 0:
        return a
    w = int(max(1, min(window, len(a))))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(a, kernel, mode="same")

def summarize_training(log) -> dict:
    """Compact metrics you might print or log to file."""
    wins, draws, loses = np.sum(log.win), np.sum(log.draw), np.sum(log.lose)
    total = max(1, len(log.episode))
    return {
        "episodes": int(total),
        "wins": int(wins),
        "draws": int(draws),
        "loses": int(loses),
        "win_rate": float(wins / total),
        "draw_rate": float(draws / total),
        "lose_rate": float(loses / total),
        "last_moving_win": float(log.moving_win[-1] if log.moving_win else 0.0),
    }

def plot_training_curves(log, window: int = 100, figsize=(12, 6)):
    """Loss curve + rolling Win/Draw/Lose rates."""
    episodes = np.asarray(log.episode, dtype=int)
    loss = np.asarray(log.loss, dtype=np.float32)
    win = np.asarray(log.win, dtype=np.float32)
    draw = np.asarray(log.draw, dtype=np.float32)
    lose = np.asarray(log.lose, dtype=np.float32)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(episodes, loss, linewidth=1)
    ax1.set_title("Policy Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(episodes, _rolling_mean(win, window), label="Win", linewidth=1.5)
    ax2.plot(episodes, _rolling_mean(draw, window), label="Draw", linewidth=1.5)
    ax2.plot(episodes, _rolling_mean(lose, window), label="Lose", linewidth=1.5)
    ax2.set_title(f"Rolling Outcome Rates (window={window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_move_preference(log, normalize: bool = True, figsize=(6, 6)):
    """Heatmap of how often each cell (0..8) was selected by the agent during training."""
    counts = np.asarray(log.move_counts, dtype=np.float32)
    grid = counts.reshape(BOARD_SIZE, BOARD_SIZE).copy()
    if normalize and counts.sum() > 0:
        grid = grid / counts.sum()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(grid, interpolation="nearest")
    ax.set_title("Move Preference Heatmap" + (" (normalized)" if normalize else " (counts)"))
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = grid[r, c]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(c, r, txt, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_training_dashboard(log, window: int = 100):
    """Dashboard = loss curve + rolling rates + normalized heatmap."""
    episodes = np.asarray(log.episode, dtype=int)
    loss = np.asarray(log.loss, dtype=np.float32)
    win = np.asarray(log.win, dtype=np.float32)
    draw = np.asarray(log.draw, dtype=np.float32)
    lose = np.asarray(log.lose, dtype=np.float32)
    counts = np.asarray(log.move_counts, dtype=np.float32)
    grid = counts.reshape(BOARD_SIZE, BOARD_SIZE)
    grid_norm = grid / grid.sum() if grid.sum() > 0 else grid

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(episodes, loss, linewidth=1)
    ax1.set_title("Policy Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(episodes, _rolling_mean(win, window), label="Win", linewidth=1.5)
    ax2.plot(episodes, _rolling_mean(draw, window), label="Draw", linewidth=1.5)
    ax2.plot(episodes, _rolling_mean(lose, window), label="Lose", linewidth=1.5)
    ax2.set_title(f"Rolling Outcome Rates (window={window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(2, 2, (3, 4))
    im = ax3.imshow(grid_norm, interpolation="nearest")
    ax3.set_title("Move Preference Heatmap (normalized)")
    ax3.set_xticks(range(BOARD_SIZE))
    ax3.set_yticks(range(BOARD_SIZE))
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            ax3.text(c, r, f"{grid_norm[r, c]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()