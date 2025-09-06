# tictactoe/game_logic.py

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .utils import BOARD_SIZE, N_ACTIONS, PLAYER_X, PLAYER_O, EMPTY

# --- Winning line indices ---
def _all_lines_indices() -> List[List[Tuple[int, int]]]:
    lines = []
    for i in range(BOARD_SIZE):
        lines.append([(i, j) for j in range(BOARD_SIZE)])                # rows
        lines.append([(j, i) for j in range(BOARD_SIZE)])                # cols
    lines.append([(i, i) for i in range(BOARD_SIZE)])                    # main diag
    lines.append([(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])   # anti diag
    return lines

WIN_LINES_RC = _all_lines_indices()
WIN_LINES_FLAT = [[r * BOARD_SIZE + c for (r, c) in line] for line in WIN_LINES_RC]

# --- Core board helpers ---
def check_winner(board: np.ndarray) -> int:
    """Return +1 if X wins, -1 if O wins, 0 otherwise."""
    b = board.reshape(BOARD_SIZE, BOARD_SIZE)
    for line in WIN_LINES_RC:
        s = sum(b[r, c] for (r, c) in line)
        if s == BOARD_SIZE * PLAYER_X:
            return PLAYER_X
        if s == BOARD_SIZE * PLAYER_O:
            return PLAYER_O
    return 0

def is_draw(board: np.ndarray) -> bool:
    return (board != EMPTY).all() and check_winner(board) == 0

def legal_actions(board: np.ndarray) -> List[int]:
    return [i for i in range(N_ACTIONS) if board[i] == EMPTY]

def legal_mask(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    mask[legal_actions(board)] = 1.0
    return mask

def apply_move(board: np.ndarray, action: int, player: int) -> np.ndarray:
    assert board[action] == EMPTY, "Illegal move"
    out = board.copy()
    out[action] = player
    return out

def encode_state(board: np.ndarray, player: int) -> torch.Tensor:
    """10-dim tensor: 9 cells in {-1,0,1} + current player as last feature."""
    x = np.concatenate([board.astype(np.float32), np.array([float(player)], dtype=np.float32)])
    return torch.from_numpy(x)

def render_board(board: np.ndarray) -> str:
    sym = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: " "}
    rows = []
    for r in range(BOARD_SIZE):
        row = " | ".join(sym[board[r * BOARD_SIZE + c]] for c in range(BOARD_SIZE))
        rows.append(f" {row} ")
    sep = "\n" + "-" * (BOARD_SIZE * 4 - 1) + "\n"
    return sep.join(rows)

# --- Easy opponent ---
def random_policy(board: np.ndarray, player: int) -> int:
    """Pick any legal move uniformly at random."""
    return int(np.random.choice(legal_actions(board)))

# --- Environment ---
@dataclass
class StepResult:
    next_state: torch.Tensor
    reward: float
    done: bool
    info: dict

class TicTacToeEnv:
    def __init__(self, start_player: int = PLAYER_X):
        self.start_player = start_player
        self.reset(start_player)

    def reset(self, start_player: Optional[int] = None) -> torch.Tensor:
        if start_player is not None:
            self.start_player = start_player
        self.board = np.zeros(N_ACTIONS, dtype=np.int8)
        self.current_player = self.start_player
        self.done = False
        self.winner = 0
        return encode_state(self.board, self.current_player)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Game is over. Call reset().")
        if self.board[action] != EMPTY:
            # illegal = immediate loss for current player
            self.done = True
            self.winner = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
            return StepResult(encode_state(self.board, self.current_player), -1.0, True, {"illegal": True})

        # apply
        self.board[action] = self.current_player

        # terminal checks
        w = check_winner(self.board)
        if w != 0:
            self.done = True
            self.winner = w
            return StepResult(encode_state(self.board, self.current_player), 1.0, True, {})

        if is_draw(self.board):
            self.done = True
            self.winner = 0
            return StepResult(encode_state(self.board, self.current_player), 0.0, True, {})

        # switch turn
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        return StepResult(encode_state(self.board, self.current_player), 0.0, False, {})

    def legal_mask(self) -> np.ndarray:
        return legal_mask(self.board)

    def render(self) -> None:
        print(render_board(self.board))