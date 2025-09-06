#!/usr/bin/env python3
"""
Tic-Tac-Toe Neural Network Demo
Demonstrates training and playing with the neural network
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from tictactoe.utils import select_device, set_seed, env_summary
from tictactoe.neural_network import PolicyNet, TrainConfig, train
from tictactoe.training import plot_training_curves, plot_training_dashboard, summarize_training

def main():
    """Run the tic-tac-toe training demo"""
    
    print("Tic-Tac-Toe Neural Network Demo")
    print("=" * 50)
    print(env_summary())
    print()
    
    # Initialize
    device = select_device()
    set_seed(42)
    
    # Train all three difficulties
    print("Training Easy (vs Random)...")
    net_easy = PolicyNet().to(device)
    cfg_easy = TrainConfig(episodes=1500, lr=1e-3, entropy_coef=0.01)
    log_easy = train(net_easy, cfg_easy)  # Fixed: was train(net, cfg)
    plot_training_curves(log_easy, window=100)
    
    print("Training Medium (vs Heuristic)...")
    net_med = PolicyNet().to(device)
    cfg_med = TrainConfig(episodes=2000, lr=1e-3, entropy_coef=0.01)
    log_med = train_vs_heuristic(net_med, cfg_med)
    plot_training_curves(log_med, window=100)  # Fixed: was plot_training_curves(log, window=100)
    
    print("Training Hard (Self-play)...")
    net_hard = PolicyNet().to(device)
    cfg_hard = TrainConfig(episodes=3000, lr=1e-3, entropy_coef=0.01)
    log_hard = train_selfplay(net_hard, cfg_hard)
    plot_training_curves(log_hard, window=100)  # Fixed: was plot_training_curves(log, window=100)
    
    # Save models
    model_path = PROJECT_ROOT / "tictactoe" / "models"
    model_path.mkdir(exist_ok=True)
    torch.save(net_easy.state_dict(), model_path / "tictactoe_easy.pt")
    torch.save(net_med.state_dict(), model_path / "tictactoe_medium.pt")
    torch.save(net_hard.state_dict(), model_path / "tictactoe_hard.pt")

if __name__ == "__main__":
    main()