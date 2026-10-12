from connect4_bitboard import Connect4, COLS, mirror_key

# Center column first, then adjacent columns
MOVE_ORDER = tuple(COLS // 2 + ((1 - 2 * (i % 2)) * ((i + 1) // 2)) for i in range(COLS))

class NegamaxSolver:
    def __init__(self, game):
        self.game = game
        self.trans_table = {}

        self.TT_HIT = 0

    def get_moves(self):
        is_valid_move = self.game.is_valid_move
        return [col for col in MOVE_ORDER if is_valid_move(col)]

    def negamax(self):
        """Entry point for the negamax search. Determines whether to use symmetric or raw negamax"""
        game = self.game
        if game.position_bb & mirror_key(game.occupied_bb ^ game.position_bb):
            print("Using raw negamax")
            return self.negamax_raw()

        print("Using symmetric negamax")
        return self.negamax_symm()

    def negamax_symm(self, alpha: int = -1, beta: int = 1):
        """Negamax search with alpha-beta pruning."""
        game = self.game

        # 1. Check for draw
        if game.is_draw():
            return 0, -1
        
        # 2. Check for transposition table hit
        cano_key, is_mirrored = game.get_canonical_key()

        tt_entry = self.trans_table.get(cano_key)
        if tt_entry is not None:
            self.TT_HIT += 1

            tt_score, tt_move = tt_entry
            board_move = COLS - 1 - tt_move if is_mirrored else tt_move
            return tt_score, board_move

        # 3. Check for kill move
        kill_move = game.get_kill_move()
        if kill_move is not None:
            cano_kill = COLS - 1 - kill_move if is_mirrored else kill_move
            self.trans_table[cano_key] = (1, cano_kill)
            return 1, kill_move

        # 4. Check for block moves
        block_moves = game.get_block_moves()
        if len(block_moves) > 1:
            self.trans_table[cano_key] = (-1, -1)
            return -1, -1

        # 5. Generate moves and search
        moves = block_moves if len(block_moves) == 1 else self.get_moves()

        best_score = -2
        best_col = -1

        for move in moves:
            # apply move and recurse
            game.make_move(move)
            score, _ = self.negamax_symm(-beta, -alpha)
            score = -score
            game.undo_move(move)

            if score > best_score:
                best_score = score
                best_col = move

            if score > alpha:
                alpha = score
            
            if alpha >= beta or best_score == 1:
                break

        cano_move = COLS - 1 - best_col if is_mirrored else best_col
        self.trans_table[cano_key] = (best_score, cano_move)

        return best_score, best_col
    
    def negamax_raw(self, alpha: int = -1, beta: int = 1):
        """Negamax search with alpha-beta pruning."""
        game = self.game
        
        # 1. Check for draw
        if game.is_draw():
            return 0, -1
        
        # 2. Check for transposition table hit
        board_key = game.get_key()
        tt_entry = self.trans_table.get(board_key)
        if tt_entry is not None:
            self.TT_HIT += 1
            return tt_entry

        # 3. Check for kill move
        kill_move = game.get_kill_move()
        if kill_move is not None:
            result = (1, kill_move)
            self.trans_table[board_key] = result
            return result

        # 4. Check for block moves
        block_moves = game.get_block_moves()
        if len(block_moves) > 1:
            result = (-1, -1)
            self.trans_table[board_key] = result
            return result

        # 5. Generate moves and search
        moves = block_moves if len(block_moves) == 1 else self.get_moves()

        best_score = -2
        best_col = -1

        for move in moves:
            # apply move and recurse
            game.make_move(move)
            score, _ = self.negamax_raw(-beta, -alpha)
            score = -score
            game.undo_move(move)

            if score > best_score:
                best_score = score
                best_col = move

            alpha = max(alpha, score)
            if alpha >= beta or best_score == 1:
                break

        result = (best_score, best_col)
        self.trans_table[board_key] = result

        return result


if __name__ == "__main__":
    import cProfile
    import pstats
    # board = [[0] * 7 for _ in range(6)]
    board = [
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [2, 0, 1, 1, 0, 0, 0],
    ]
    game = Connect4()
    game.from_board(board)
    print("=== Loaded Board ===")
    print(game)
    print(f"Player to move: {game.player_label()}")

    solver = NegamaxSolver(game)
    
    # cProfile the negamax search
    profiler = cProfile.Profile()
    profiler.enable()

    score, best_move = solver.negamax()

    profiler.disable()

    print("\n=== Solver Result ===")
    print(f"Predicted Score: {score}")
    print(f"Optimal Move: {best_move}")
    print(f"TT size: {len(solver.trans_table)}")
    print(f"TT Hits: {solver.TT_HIT}")
    print(f"TT Hit Rate: {solver.TT_HIT / len(solver.trans_table) * 100:.2f}%")

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("tottime").print_stats(20)