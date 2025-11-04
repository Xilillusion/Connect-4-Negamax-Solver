import numpy as np
import multiprocessing as mp
import os
from functools import lru_cache

ZOBRIST_TABLE = None
HASH_MASK = None
TRANS_TABLE = None
BASE_BOARD = None
CONNECT_N = None
ROOT_ALPHA = None
ROOT_BETA = None


@lru_cache(maxsize=None)
def _generate_zobrist(rows: int, cols: int):
    rng = np.random.default_rng(42 + rows * 31 + cols)
    table = rng.integers(0, 1 << 63, size=(rows, cols, 2), dtype=np.uint64)
    mask = (1 << 63) - 1
    return table, mask


def init_pool(zobrist_table, hash_mask, trans_table, board_state, connect, alpha, beta):
    global ZOBRIST_TABLE, HASH_MASK, TRANS_TABLE, BASE_BOARD, CONNECT_N, ROOT_ALPHA, ROOT_BETA
    ZOBRIST_TABLE = np.asarray(zobrist_table, dtype=np.uint64)
    HASH_MASK = int(hash_mask)
    TRANS_TABLE = trans_table
    BASE_BOARD = np.array(board_state, dtype=np.int8, copy=True)
    CONNECT_N = connect
    ROOT_ALPHA = alpha
    ROOT_BETA = beta


def evaluate_root(move):
    y, x = move
    solver = Connect4Solver(
        BASE_BOARD,
        CONNECT_N,
        trans_table=TRANS_TABLE,
        zobrist_table=ZOBRIST_TABLE,
        hash_mask=HASH_MASK,
    )
    solver.make_move(y, x)
    opp_score, _ = solver.negamax(-ROOT_BETA, -ROOT_ALPHA, last_move=(y, x))
    return -opp_score, x

class Connect4Solver:
    def __init__(self, board, connect: int = 4, trans_table=None, zobrist_table=None, hash_mask=None):
        self.board = np.array(board, dtype=np.int8, copy=True)
        self.player = 1 + np.count_nonzero(self.board) % 2
        self.connect = connect
        self.trans_table = trans_table if trans_table is not None else {}

        self.rows, self.cols = self.board.shape
        self.column_order = sorted(range(self.cols), key=lambda c: abs(c - self.cols // 2))
        self.top_empty = []
        for c in range(self.cols):
            col = self.board[:, c]
            empties = np.where(col == 0)[0]
            self.top_empty.append(int(empties[-1]) if empties.size else -1)

        self.init_hash(zobrist_table, hash_mask)

    def init_hash(self, zobrist_table=None, hash_mask=None):
        if zobrist_table is None or hash_mask is None:
            table, mask = _generate_zobrist(self.rows, self.cols)
        else:
            table, mask = zobrist_table, hash_mask

        self.zobrist_table = np.asarray(table, dtype=np.uint64)
        self.hash_mask = int(mask)

        # Compute initial hash and its vertical symmetry
        self.curr_hash = 0
        self.curr_symm_hash = 0
        for (r, c), val in np.ndenumerate(self.board):
            if val != 0:
                piece = int(self.zobrist_table[r, c, val - 1])
                sym_piece = int(self.zobrist_table[r, self.cols - 1 - c, val - 1])
                self.curr_hash ^= piece
                self.curr_symm_hash ^= sym_piece
        self.curr_hash &= self.hash_mask
        self.curr_symm_hash &= self.hash_mask

    def get_init_score(self):
        """
        Evaluate the initial board state for a win/loss/draw.
        Returns 1 if current player wins, -1 if opponent wins, 0 for draw, None if ongoing.
        """
        def check_line(line):
          # Find if there are winning consecutive pieces in a line
          diff = np.diff(line)
          idx = np.where(diff != 0)[0] + 1
          runs = np.split(line, idx)
          
          for run in runs:
              k = run[0]
              # Return score if found
              if k > 0 and len(run) >= self.connect:
                  return 1 if k == self.player else -1

          return None
    
        n, m = self.board.shape
        # Check rows
        for r in range(n):
            score = check_line(self.board[r, :])
            if score is not None:
                return score
            
        # Check columns
        for c in range(m):
            score = check_line(self.board[:, c])
            if score is not None:
                return score
            
        # Check diagonals and anti-diagonals
        flipped_board = np.fliplr(self.board)
        for offset in range(-n+1, m):
            line = self.board.diagonal(offset=offset)
            score = check_line(line)
            if score is not None:
                return score
            
            line = flipped_board.diagonal(offset=offset)
            score = check_line(line)
            if score is not None:
                return score

        if all(t == -1 for t in self.top_empty):
            return 0
        return None
    
    def get_score(self, last_move):
        """
        Evaluate the board state after the last move for a win/loss/draw.
        Returns 1 if current player wins, -1 if opponent wins, 0 for draw, None if ongoing.
        """
        if last_move is None:
            return self.get_init_score()

        y, x = last_move
        last_player = 3 - self.player
        
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            
            # Count in positive direction
            ny, nx = y + dy, x + dx
            while 0 <= ny < self.rows and 0 <= nx < self.cols and self.board[ny][nx] == last_player:
                count += 1
                ny += dy
                nx += dx
            # Count in negative direction
            ny, nx = y - dy, x - dx
            while 0 <= ny < self.rows and 0 <= nx < self.cols and self.board[ny][nx] == last_player:
                count += 1
                ny -= dy
                nx -= dx

            if count >= self.connect:    # check if we have a winning line
                return 1 if last_player == self.player else -1
            
        if all(t == -1 for t in self.top_empty):
            return 0
        return None
    
    def get_moves(self):
        """
        Generate possible moves ordered by heuristic scores.
        Returns a list of (y, x) tuples for valid moves.
        """
        moves = []
        for x in self.column_order:
            y = self.top_empty[x]
            if y >= 0:
                moves.append((y, x))
        return moves
    
    def make_move(self, y, x):
        """Make a move on the board."""
        self.board[y][x] = self.player
        self.top_empty[x] = y - 1

        # Update hash
        self.curr_hash ^= int(self.zobrist_table[y, x, self.player - 1])
        self.curr_hash &= self.hash_mask
        self.curr_symm_hash ^= int(self.zobrist_table[y, self.cols - 1 - x, self.player - 1])
        self.curr_symm_hash &= self.hash_mask
        
        self.player = 3 - self.player

    def undo_move(self, y, x):
        """Undo a move on the board."""
        self.board[y][x] = 0
        self.top_empty[x] = y

        self.player = 3 - self.player

        # Update hash
        self.curr_hash ^= int(self.zobrist_table[y, x, self.player - 1])
        self.curr_hash &= self.hash_mask
        self.curr_symm_hash ^= int(self.zobrist_table[y, self.cols - 1 - x, self.player - 1])
        self.curr_symm_hash &= self.hash_mask

    def negamax(self, alpha=-1, beta=1, last_move=None):
        """
        Negamax search with alpha-beta pruning, transposition table and symmetry handling.
        Returns (score, best_move) where score is in {-1, 0, 1} and best_move is the column index.
        """
        hash = self.curr_hash
        if hash in self.trans_table:
            return self.trans_table[hash]

        score = self.get_score(last_move)
        if score is not None:
            result = (score, None)
            self.trans_table[hash] = result
            self.trans_table[self.curr_symm_hash] = result
            return score, None

        best_score = -2
        best_col = None
        for move in self.get_moves():
            y, x = move
            self.make_move(y, x)
            opp_score, _ = self.negamax(-beta, -alpha, move)
            score = -opp_score
            self.undo_move(y, x)

            if score > best_score:
                best_score = score
                best_col = x

            # Update alpha-beta bounds
            alpha = max(alpha, score)
            if alpha >= beta or score == 1:
                symm_col = None if best_col is None else self.cols - 1 - best_col
                self.trans_table[hash] = (best_score, best_col)
                self.trans_table[self.curr_symm_hash] = (best_score, symm_col)
                return best_score, best_col

        symm_col = None if best_col is None else self.cols - 1 - best_col
        self.trans_table[hash] = (best_score, best_col)
        self.trans_table[self.curr_symm_hash] = (best_score, symm_col)
        return best_score, best_col

    def negamax_parallel(self, alpha=-1, beta=1, last_move=None, processes=None):
        """Parallel Negamax with shared transposition table and center-first move ordering."""
        hash_key = self.curr_hash
        if hash_key in self.trans_table:
            return self.trans_table[hash_key]

        score = self.get_score(last_move)
        if score is not None:
            result = (score, None)
            self.trans_table[hash_key] = result
            self.trans_table[self.curr_symm_hash] = result
            return result

        moves = self.get_moves()
        if not moves:
            result = (0, None)
            self.trans_table[hash_key] = result
            self.trans_table[self.curr_symm_hash] = result
            return result

        available = mp.cpu_count() if processes is None else max(1, min(processes, mp.cpu_count()))
        worker_count = min(len(moves), available)
        if worker_count <= 1:
            return self.negamax(alpha, beta, last_move)

        board_state = self.board.copy()
        zobrist_table = self.zobrist_table.copy()

        with mp.Pool(
            processes=worker_count,
            initializer=init_pool,
            initargs=(
                zobrist_table,
                self.hash_mask,
                self.trans_table,
                board_state,
                self.connect,
                alpha,
                beta,
            ),
        ) as pool:
            best_score = -2
            best_col = None
            for score, col in pool.imap_unordered(evaluate_root, moves):
                if score > best_score:
                    best_score = score
                    best_col = col

        result = (best_score, best_col)
        symm_col = None if best_col is None else self.cols - 1 - best_col
        self.trans_table[hash_key] = result
        self.trans_table[self.curr_symm_hash] = (best_score, symm_col)
        return result


def show_result(solver, score, move):
    if score == 1:
        winner = f"Player {solver.player}"
    elif score == -1:
        winner = f"Player {3-solver.player}"
    else:
        winner = "Draw"
    print(f"{solver.__class__.__name__} Result")
    print(f"Winner: {winner}, Best move for Player {solver.player}: {move}")


if __name__ == "__main__":
    mp.freeze_support()

    board = [
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 1, 0, 2, 1, 1, 0],
        [0, 2, 0, 1, 2, 2, 0],
        [0, 1, 0, 2, 1, 1, 0],
        [2, 1, 0, 1, 2, 2, 0]
    ]
    num_processes = 100
    connect = 4

    max_processes = max(1, min(num_processes, mp.cpu_count()))
    if max_processes > 1:
        manager = mp.Manager()
        try:
            trans_table = manager.dict()
            solver = Connect4Solver(board, connect=connect, trans_table=trans_table)
            score, move = solver.negamax_parallel(processes=max_processes)
        finally:
            manager.shutdown()
    else:
        solver = Connect4Solver(board, connect=connect)
        score, move = solver.negamax()

    show_result(solver, score, move)