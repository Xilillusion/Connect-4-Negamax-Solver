import numpy as np
import random
import multiprocessing as mp
import os

ZOBRIST_TABLE = None
TRANS_TABLE = None
BASE_BOARD = None
CONNECT_N = None
TASK_QUEUE = None
RESULT_QUEUE = None
ROOT_ALPHA = None
ROOT_BETA = None


def init_pool(zobrist_table, trans_table, board_state, connect, task_queue, result_queue, alpha, beta):
    global ZOBRIST_TABLE, TRANS_TABLE, BASE_BOARD, CONNECT_N, TASK_QUEUE, RESULT_QUEUE, ROOT_ALPHA, ROOT_BETA
    ZOBRIST_TABLE = zobrist_table
    TRANS_TABLE = trans_table
    BASE_BOARD = board_state
    CONNECT_N = connect
    TASK_QUEUE = task_queue
    RESULT_QUEUE = result_queue
    ROOT_ALPHA = alpha
    ROOT_BETA = beta

def worker_process_root():
    while True:
        task = TASK_QUEUE.get()
        if task is None:
            break

        y, x = task
        solver = Connect4Solver(
            BASE_BOARD,
            CONNECT_N,
            trans_table=TRANS_TABLE,
            zobrist_table=ZOBRIST_TABLE,
        )
        solver.make_move(y, x)
        opp_score, _ = solver.negamax(-ROOT_BETA, -ROOT_ALPHA, last_move=(y, x))
        score = -opp_score
        RESULT_QUEUE.put((score, x))

class Connect4Solver:
    def __init__(self, board: list, connect: int = 4, trans_table=None, zobrist_table=None):
        self.board = np.array(board)
        self.player = 1 + np.count_nonzero(self.board) % 2
        self.connect = connect
        self.trans_table = trans_table if trans_table is not None else {}

        self.rows, self.cols = self.board.shape
        self.top_empty = [self.rows-1-np.argmax(self.board[::-1,c] == 0) for c in range(self.cols)]   # -1 if column is full

        self.init_hash(zobrist_table)

    def init_hash(self, zobrist_table):
        if zobrist_table is None:
            hash_bits = int(self.rows * self.cols * 1.585) + 1
            random.seed(42)     # Tradition 
            self.zobrist_table = [[[random.getrandbits(hash_bits) for _ in range(2)]    # two players
                                   for _ in range(self.cols)]
                                   for _ in range(self.rows)]
        else:
            self.zobrist_table = zobrist_table

        # Compute initial hash and its vertical symmetry
        self.curr_hash = 0
        self.curr_symm_hash = 0
        for (r, c), val in np.ndenumerate(self.board):
            if val != 0:
                self.curr_hash ^= self.zobrist_table[r][c][val - 1]
                self.curr_symm_hash ^= self.zobrist_table[r][self.cols-1-c][val - 1]

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
        if all(t == -1 for t in self.top_empty):
            return []
        
        scores = np.zeros(len(self.top_empty), dtype=int)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def count_dir(y, x, dx, dy, player):
            length = 0
            ny, nx = y + dy, x + dx
            while 0 <= ny < self.rows and 0 <= nx < self.cols and self.board[ny][nx] == player:
                length += 1
                ny += dy
                nx += dx
            return length

        for x, y in enumerate(self.top_empty):
            if y == -1:
                scores[x] = -1
                continue

            score = 0
            for dx, dy in directions:
                for player in (1, 2):
                    total = count_dir(y, x, dx, dy, player) + count_dir(y, x, -dx, -dy, player)
                    if total == 0:
                        continue
                    score += min(total, self.connect) ** 2
            scores[x] = score

        order = np.argsort(-scores)
        moves = []
        for i in order:
            if self.top_empty[i] >= 0:  # valid column
                moves.append((self.top_empty[i], i))  # (y, x)
        
        return moves
    
    def make_move(self, y, x):
        """Make a move on the board."""
        self.board[y][x] = self.player
        self.top_empty[x] -= 1

        # Update hash
        self.curr_hash ^= self.zobrist_table[y][x][self.player - 1]
        self.curr_symm_hash ^= self.zobrist_table[y][self.cols-1-x][self.player-1]
        
        self.player = 3 - self.player

    def undo_move(self, y, x):
        """Undo a move on the board."""
        self.board[y][x] = 0
        self.top_empty[x] += 1

        self.player = 3 - self.player

        # Update hash
        self.curr_hash ^= self.zobrist_table[y][x][self.player - 1]
        self.curr_symm_hash ^= self.zobrist_table[y][self.cols-1-x][self.player-1]

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

        best_score = -1
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
                self.trans_table[hash] = (best_score, best_col)
                self.trans_table[self.curr_symm_hash] = (best_score, self.cols-1-best_col)  # symmetric column
                return best_score, best_col

        self.trans_table[hash] = (best_score, None)
        self.trans_table[self.curr_symm_hash] = (best_score, None)  
        return best_score, None

    def negamax_parallel(self, alpha=-1, beta=1, last_move=None, processes=mp.cpu_count()):
        """Parallel Negamax that uses a task queue for dynamic root move scheduling."""
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
        worker_count = min(len(moves), processes)
        if worker_count <= 1:
            return self.negamax(alpha, beta, last_move)

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        for move in moves:
            task_queue.put(move)
        for _ in range(worker_count):
            task_queue.put(None)

        board_state = self.board.tolist()
        zobrist_table = self.zobrist_table

        with mp.Pool(
            processes=worker_count,
            initializer=init_pool,
            initargs=(
                zobrist_table,
                self.trans_table,
                board_state,
                self.connect,
                task_queue,
                result_queue,
                alpha,
                beta,
            ),
        ) as pool:
            workers = [pool.apply_async(worker_process_root) for _ in range(worker_count)]

            best_score = -1
            best_col = None
            for _ in range(len(moves)):
                score, col = result_queue.get()
                if score > best_score:
                    best_score = score
                    best_col = col

            for worker in workers:
                worker.get()

        for q in (task_queue, result_queue):
            try:
                q.close()
                q.join_thread()
            except (AttributeError, OSError):
                pass

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
        [0, 1, 0, 1, 2, 0, 2],
        [0, 2, 0, 1, 2, 0, 1],
        [0, 1, 0, 2, 1, 1, 2],
        [0, 2, 0, 1, 2, 2, 1],
        [0, 1, 0, 2, 1, 1, 2],
        [2, 1, 0, 1, 2, 2, 1]
    ]
    num_processes = 100
    connect = 4
    
    max_processes = max(1, min(num_processes, mp.cpu_count()))
    if max_processes > 1:
        manager = mp.Manager()
        solver = Connect4Solver(board, connect=connect, trans_table=manager.dict())
        score, move = solver.negamax_parallel(processes=max_processes)
    else:
        shared_trans = {}
    
    show_result(solver, score, move)