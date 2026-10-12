import numpy as np
from .connect4_array import Connect4

class NegamaxSolver:
    def __init__(self, game, seed: int = 42):
        self.game = game
        self.trans_table = {}
        self.rows, self.cols = game.rows, game.cols
        self.column_order = sorted(range(self.cols), key=lambda c: abs(c - self.cols // 2))

        # Current incremental hashes for the board and its vertical symmetry
        self.hash = np.uint64(0)
        self.symm_hash = np.uint64(0)

        # Zobrist table: one 64-bit random integer per (row, col, player)
        rng = np.random.default_rng(seed + self.rows * 31 + self.cols)
        self.zobrist_table = rng.integers(0, 2**64, size=(self.rows, self.cols, 2), dtype=np.uint64)

        # Initialize hashes from the starting board
        for (r, c), val in np.ndenumerate(self.game.board):
            v = int(val)
            if v != 0:
                self.hash ^= self.zobrist_table[r, c, v - 1]
                self.symm_hash ^= self.zobrist_table[r, self.cols - 1 - c, v - 1]
    
    def update_hash(self, r: int, c: int, player_idx: int):
        """
        Toggle Zobrist for a piece placed at (r,c) by `player` (1 or 2).
        This updates both the board-hash and the symmetric-board-hash.
        """
        self.hash ^= self.zobrist_table[r, c, player_idx]
        self.symm_hash ^= self.zobrist_table[r, self.cols-1-c, player_idx]
    
    def canonical_hash(self):
        return min(int(self.hash), int(self.symm_hash)), self.hash > self.symm_hash

    def get_moves(self):
        """
        Generate possible moves ordered by heuristic scores.
        Returns a list of column indices for valid moves.
        """
        blocking = []
        neutral = []
        for x in self.column_order:
            y = self.game.heights[x]
            if y < 0:
                continue
            
            player = self.game.player
            if self.game.check_line(y, x, player):
                return [x]  # Winning move, immediate cutoff
            elif self.game.check_line(y, x, 3 - player):
                blocking.append(x)  # Blocking move
            else:
                neutral.append(x)  # Neutral move
        
        return blocking + neutral

    def negamax(self, alpha: int, beta: int, last_move=None, depth: int = 0):
        """
        Negamax search with alpha-beta pruning.
        Uses incremental Zobrist hashes stored in `self.hash` and `self.symm_hash`.
        """
        board_key, is_symm = self.canonical_hash()
        if board_key in self.trans_table:
            score, col = self.trans_table[board_key]
            return score, self.cols-1-col if is_symm and col is not None else col

        score = self.game.get_score(last_move)
        if score is not None:
            result = (score, None)
            # store for both normal and symmetric hashes
            self.trans_table[board_key] = result
            return result

        best_score = -2
        best_col = None

        for move in self.get_moves():
            # y is the row index the piece will occupy (top_empty before move)
            y = int(self.game.heights[move])
            player_idx = self.game.player - 1

            # apply move and recurse
            self.game.make_move(move)
            self.update_hash(y, move, player_idx)
            score, _ = self.negamax(-beta, -alpha, move, depth+1)
            score = -score

            # undo move and revert Zobrist
            self.update_hash(y, move, player_idx)
            self.game.undo_move(move)

            if score > best_score:
                best_score = score
                best_col = move

            alpha = max(alpha, score)
            if alpha >= beta or best_score == 1:
                break

        board_key, is_symm = self.canonical_hash()
        canonical_col = best_col if not is_symm or best_col is None else self.cols-1-best_col
        self.trans_table[board_key] = (best_score, canonical_col)
        return best_score, best_col


class IDABSolver(NegamaxSolver):
    def negamax(self, alpha: int, beta: int, last_move=None, max_depth: int = 0):
        """
        Negamax search with alpha-beta pruning.
        Uses incremental Zobrist hashes stored in `self.hash` and `self.symm_hash`.
        """
        if max_depth == 0:
            return 0, None  # treat non-terminal leaf nodes as neutral

        pass

    def idab(self):
        best_move = None

        for depth in range(self.max_depth):

            if best_move is not None:
                self.game.get_moves().sort(key=lambda m: m == best_move, reverse=True)

            score, best_move = self.negamax(-1, 1, None, depth)

        return score, best_move

class StatNegamaxSolver(NegamaxSolver):
    def __init__(self, game, seed: int = 42):
        super().__init__(game, seed)
        from app.visualizer import SearchStats

        self.stats = SearchStats(self.cols)

    def negamax(self, alpha: int = -1, beta: int = 1, last_move=None, depth: int = 0):
        """
        Negamax search with alpha-beta pruning and statistics collection.
        """
        board_key, is_symm = self.canonical_hash()

        is_hit = board_key in self.trans_table
        self.stats.update(depth, last_move, is_hit)

        if is_hit:
            score, col = self.trans_table[board_key]
            return score, self.cols-1-col if is_symm and col is not None else col

        pass

if __name__ == "__main__":
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    game = Connect4(board)
    solver = NegamaxSolver(game)
    score, best_move = solver.negamax(42, -42)
    print(f"Best move: {best_move}, Score: {score}")
    print("Transpositions:", len(solver.trans_table))
