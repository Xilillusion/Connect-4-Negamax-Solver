import numpy as np
import random

class Connect4Solver:
    def __init__(self, board: list, connect: int=4):
        self.board = np.array(board)
        self.player = 1 + np.count_nonzero(self.board) % 2
        self.connect = connect

        self.trans_table = {} # {hash_key: (best_score, best_col)}

        self.rows, self.cols = self.board.shape
        self.top_empty = [self.rows-1-np.argmax(self.board[::-1,c] == 0) for c in range(self.cols)]

        self.init_hash()

        self.explored_count = 0
        self.trans_count = 0
        self.pruning_count = 0

    def show_counts(self):
        print(f"Explored: {self.explored_count}, Pruning: {self.pruning_count}, TT Cutoff: {self.trans_count}")

    def init_hash(self):
        # Initialize Zobrist hashing
        # Calculate minimum bits needed: log2(3^(rows*cols)) ≈ rows*cols*1.585
        hash_bits = int(self.rows * self.cols * 1.585) + 1
        self.hash_mask = (1 << hash_bits) - 1

        # Generate Zobrist table [row][col][player-1]
        random.seed(42)
        self.zobrist_table = [[[random.getrandbits(hash_bits) for _ in range(2)] 
                              for _ in range(self.cols)] 
                             for _ in range(self.rows)]
        
        # Calculate initial hash and symmetric hash
        self.curr_hash = 0
        self.curr_symm_hash = 0
        for (r, c), val in np.ndenumerate(self.board):
            if val != 0:
                self.curr_hash ^= self.zobrist_table[r][c][val - 1]
                self.curr_symm_hash ^= self.zobrist_table[r][self.cols-1-c][val - 1]
        self.curr_hash &= self.hash_mask
        self.curr_symm_hash &= self.hash_mask

    def get_init_score(self):
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
        # Check if last move caused a win
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
        # Return list of valid moves ordered by the number of potential connections
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
        # Make a move and update the hash
        self.board[y][x] = self.player
        self.top_empty[x] -= 1

        self.curr_hash ^= self.zobrist_table[y][x][self.player - 1]
        self.curr_hash &= self.hash_mask
        
        # Update symmetric hash
        self.curr_symm_hash ^= self.zobrist_table[y][self.cols-1-x][self.player-1]
        self.curr_symm_hash &= self.hash_mask
        
        self.player = 3 - self.player

    def undo_move(self, y, x):
        # Undo a move and update the hash
        self.board[y][x] = 0
        self.top_empty[x] += 1

        self.player = 3 - self.player

        self.curr_hash ^= self.zobrist_table[y][x][self.player - 1]
        self.curr_hash &= self.hash_mask
        
        # Update symmetric hash
        self.curr_symm_hash ^= self.zobrist_table[y][self.cols-1-x][self.player-1]
        self.curr_symm_hash &= self.hash_mask
    
    def negamax(self, alpha=-1, beta=1, last_move=None):
        # Negamax search with alpha-beta pruning, transposition table and symmetry handling
        hash = self.curr_hash
        if hash in self.trans_table:
            self.trans_count += 1
            return self.trans_table[hash]

        score = self.get_score(last_move)
        if score is not None:
            result = (score, None)
            self.trans_table[hash] = result
            self.trans_table[self.curr_symm_hash] = result
            return score, None

        best_score = -1
        for move in self.get_moves():
            self.explored_count += 1
            if self.explored_count % 1000000 == 0:
                self.show_counts()

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
                self.pruning_count += 1

                self.trans_table[hash] = (best_score, best_col)
                self.trans_table[self.curr_symm_hash] = (best_score, self.cols-1-best_col)  # symmetric column
                return best_score, best_col

        self.trans_table[hash] = (best_score, None)
        self.trans_table[self.curr_symm_hash] = (best_score, None)  
        return best_score, None

def show_result(solver, score, move):
    if score == 1:
        winner = f"Player {solver.player}"
    elif score == -1:
        winner = f"Player {3-solver.player}"
    else:
        winner = "Draw"
    print(f"{solver.__class__.__name__} Result")
    solver.show_counts(solver)
    print(f"Winner: {winner}, Best move for Player {solver.player}: {move}")

if __name__ == "__main__":
    board = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    solver = Connect4Solver(board, connect=4)
    score, move = solver.negamax()
    
    show_result(solver, score, move)