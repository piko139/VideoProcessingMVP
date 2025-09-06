# tictactoe/neural_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from collections import deque

from .utils import DEVICE, N_ACTIONS, PLAYER_X, PLAYER_O
from .game_logic import TicTacToeEnv, encode_state, random_policy

# --- Policy network ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int = 10, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits for 9 actions (unmasked)

# --- Masking & action selection ---
def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(mask <= 0, neg_inf)

@torch.no_grad()
def greedy_action(policy: PolicyNet, state: torch.Tensor, mask_np: np.ndarray) -> int:
    x = state.to(DEVICE).float().unsqueeze(0)
    logits = policy(x).squeeze(0)
    mask = torch.from_numpy(mask_np).to(DEVICE).float()
    masked = _mask_logits(logits, mask)
    return int(torch.argmax(masked).item())

def sample_action(policy: PolicyNet, state: torch.Tensor, mask_np: np.ndarray, temperature: float = 1.0):
    x = state.to(DEVICE).float().unsqueeze(0)
    logits = policy(x).squeeze(0)
    if temperature != 1.0:
        logits = logits / float(temperature)
    mask = torch.from_numpy(mask_np).to(DEVICE).float()
    masked = _mask_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    return int(a.item()), dist.log_prob(a), dist.entropy()

# --- Configs & logs ---
@dataclass
class TrainConfig:
    episodes: int = 2000
    lr: float = 1e-3
    entropy_coef: float = 0.01
    temperature: float = 1.0
    eval_every: int = 200
    eval_games: int = 200
    start_player_mix: bool = True

@dataclass
class TrainLog:
    episode: List[int]
    loss: List[float]
    entropy: List[float]
    win: List[int]
    draw: List[int]
    lose: List[int]
    moving_win: List[float]
    move_counts: np.ndarray
    notes: dict

# --- Evaluation (greedy vs Random) ---
def evaluate(policy: PolicyNet, games: int = 200) -> Tuple[float, float, float]:
    env = TicTacToeEnv()
    wins = draws = losses = 0
    for g in range(games):
        start = PLAYER_X if (g % 2 == 0) else PLAYER_O
        env.reset(start)
        while not env.done:
            if env.current_player == start:
                a = greedy_action(policy, encode_state(env.board, env.current_player), env.legal_mask())
            else:
                a = random_policy(env.board.copy(), env.current_player)
            env.step(a)
        if env.winner == 0:
            draws += 1
        elif env.winner == start:
            wins += 1
        else:
            losses += 1
    total = float(games)
    return wins / total, draws / total, losses / total

# --- Training (REINFORCE) vs Random ---
def train(policy: PolicyNet, cfg: TrainConfig) -> TrainLog:
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    env = TicTacToeEnv()
    move_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    window = deque(maxlen=200)

    log = TrainLog(
        episode=[], loss=[], entropy=[], win=[], draw=[], lose=[], 
        moving_win=[], move_counts=move_counts, notes={}
    )

    for ep in range(1, cfg.episodes + 1):
        start = PLAYER_X if (cfg.start_player_mix and (ep % 2 == 0)) else PLAYER_O
        env.reset(start)

        log_probs_agent: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        while not env.done:
            s = encode_state(env.board, env.current_player)
            mask_np = env.legal_mask()

            if env.current_player == start:
                a, lp, ent = sample_action(policy, s, mask_np, temperature=cfg.temperature)
                move_counts[a] += 1
                log_probs_agent.append(lp)
                entropies.append(ent)
            else:
                a = random_policy(env.board.copy(), env.current_player)

            env.step(a)

        if env.winner == start:
            r = 1.0
            window.append(1.0)
        elif env.winner == 0:
            r = 0.0
            window.append(0.5)
        else:
            r = -1.0
            window.append(0.0)

        if log_probs_agent:
            pg_loss = -(r * torch.stack(log_probs_agent).sum())
        else:
            pg_loss = torch.tensor(0.0, device=DEVICE)

        ent_term = -cfg.entropy_coef * (torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=DEVICE))
        loss = pg_loss + ent_term

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        log.episode.append(ep)
        log.loss.append(float(loss.detach().cpu().item()))
        log.entropy.append(float((torch.stack(entropies).mean() if entropies else torch.tensor(0.0)).cpu().item()))
        log.win.append(1 if env.winner == start else 0)
        log.draw.append(1 if env.winner == 0 else 0)
        log.lose.append(1 if (env.winner != 0 and env.winner != start) else 0)
        log.moving_win.append(float(sum(window) / len(window)) if len(window) else 0.0)

        if (ep % cfg.eval_every == 0) or (ep == cfg.episodes):
            wr, dr, lr = evaluate(policy, games=min(cfg.eval_games, 200))
            log.notes[f"eval@{ep}"] = f"vs Random — Win {wr:.2f}, Draw {dr:.2f}, Lose {lr:.2f}"

    log.move_counts = move_counts
    return log