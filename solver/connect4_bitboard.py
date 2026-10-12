ROWS = 6
COLS = 7
STRIDE = ROWS + 1  # 7

# Mask covering only row 0
BOTTOM_MASK = sum(1 << (c * STRIDE) for c in range(COLS))

# Mask covering all 42 valid board positions
BOARD_MASK = sum(((1 << ROWS) - 1) << (c * STRIDE) for c in range(COLS))

# Mask covering the top playable cell of each column (Row 5)
TOP_ROW_BITS = tuple((1 << (ROWS - 1)) << (c * STRIDE) for c in range(COLS))

# Column masks for O(1) branchless horizontal key mirroring
_COL_MASK = (1 << STRIDE) - 1  # 0x7F
_C0 = _COL_MASK
_C1 = _COL_MASK << 7
_C2 = _COL_MASK << 14
_C3 = _COL_MASK << 21
_C4 = _COL_MASK << 28
_C5 = _COL_MASK << 35
_C6 = _COL_MASK << 42


def mirror_key(key):
    """Reflects a John Tromp position key horizontally across the center column.
    Swaps columns (0 <-> 6), (1 <-> 5), (2 <-> 4) in ~6 bitwise operations.
    """
    return (
        ((key & _C0) << 42)
        | ((key & _C6) >> 42)
        | ((key & _C1) << 28)
        | ((key & _C5) >> 28)
        | ((key & _C2) << 14)
        | ((key & _C4) >> 14)
        | (key & _C3)
    )

class Connect4:
    def __init__(self):
        self.position_bb = 0
        self.occupied_bb = 0
        self.player_idx = 0
        self.empt_count = ROWS * COLS
        self.heights = [c * STRIDE for c in range(COLS)]

    def __str__(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        res = []
        for row in self.to_board():
            res.append(' '.join(symbols[int(cell)] for cell in row))
        res.append("-" * (COLS * 2 - 1))
        res.append(' '.join(str(i) for i in range(COLS)))
        return "\n".join(res)

    def player_label(self, player_idx=None):
        if player_idx is None:
            player_idx = self.player_idx
        return f"Player {player_idx + 1} ({'X' if player_idx == 0 else 'O'})"

    def to_board(self):
        """Converts the bitboard representation to a 2D list representation of the board."""
        board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        p1_bb = (
            self.position_bb
            if self.player_idx == 0
            else (self.position_bb ^ self.occupied_bb)
        )
        p2_bb = self.occupied_bb ^ p1_bb

        for c in range(COLS):
            for r in range(ROWS):
                mask = 1 << (c * STRIDE + r)
                if p1_bb & mask:
                    board[ROWS - 1 - r][c] = 1
                elif p2_bb & mask:
                    board[ROWS - 1 - r][c] = 2
        return board

    def from_board(self, board):
        """Initializes the bitboard representation from a 2D list representation of the board."""
        p1_bb = 0
        p2_bb = 0
        self.heights = [c * STRIDE for c in range(COLS)]

        ply_count = 0

        for col in range(COLS):
            row_idx = 0
            for rev_row in reversed(range(ROWS)):
                val = board[rev_row][col]
                if val == 0:
                    break   # Stop at the first empty cell in the column

                bit = 1 << (col * STRIDE + row_idx)
                if val == 1:
                    p1_bb |= bit
                elif val == 2:
                    p2_bb |= bit

                row_idx += 1
                ply_count += 1

            self.heights[col] = col * STRIDE + row_idx

        self.occupied_bb = p1_bb | p2_bb
        self.empt_count = ROWS * COLS - ply_count
        self.player_idx = ply_count & 1
        self.position_bb = p1_bb if self.player_idx == 0 else p2_bb

    def is_valid_move(self, col):
        """Checks if a move in the specified column is valid."""
        return not (self.occupied_bb & TOP_ROW_BITS[col])

    def winning_positions(self, player_bb):
        """Returns a bitboard of all winning positions for the given player's bitboard."""
        r = (player_bb << 1) & (player_bb << 2) & (player_bb << 3)

        # Horizontal
        p = (player_bb << 7) & (player_bb << 14)
        r |= p & (player_bb << 21)
        r |= p & (player_bb >> 7)
        p = (player_bb >> 7) & (player_bb >> 14)
        r |= p & (player_bb << 7)
        r |= p & (player_bb >> 21)

        # Diagonal
        p = (player_bb << 8) & (player_bb << 16)
        r |= p & (player_bb << 24)
        r |= p & (player_bb >> 8)
        p = (player_bb >> 8) & (player_bb >> 16)
        r |= p & (player_bb << 8)
        r |= p & (player_bb >> 24)

        # Anti-diagonal
        p = (player_bb << 6) & (player_bb << 12)
        r |= p & (player_bb << 18)
        r |= p & (player_bb >> 6)
        p = (player_bb >> 6) & (player_bb >> 12)
        r |= p & (player_bb << 6)
        r |= p & (player_bb >> 18)

        return (r & BOARD_MASK) & (BOARD_MASK ^ self.occupied_bb)
    
    def get_kill_move(self):
        """Returns the column index of a winning move for the current player, if one exists. Otherwise, returns None."""
        curr_threats = self.winning_positions(self.position_bb)
        if not curr_threats:
            return None

        legal_moves = (self.occupied_bb + BOTTOM_MASK) & BOARD_MASK
        win_spots = legal_moves & curr_threats

        if not win_spots:
            return None

        # If there are multiple winning spots, return the first one found
        lsb = win_spots & -win_spots
        bit_index = lsb.bit_length() - 1
        return bit_index // STRIDE

    def get_block_moves(self):
        """Returns a list of column indices that block the opponent's winning moves. If there are no immediate threats, returns an empty list."""
        opp_bb = self.occupied_bb ^ self.position_bb
        opp_threats = self.winning_positions(opp_bb)
        if not opp_threats:
            return []

        legal_moves = (self.occupied_bb + BOTTOM_MASK) & BOARD_MASK
        block_spots = legal_moves & opp_threats

        # If there are multiple blocking spots, return all of them
        blocking_cols = []
        while block_spots:
            lsb = block_spots & -block_spots
            bit_index = lsb.bit_length() - 1
            blocking_cols.append(bit_index // STRIDE)
            block_spots ^= lsb

        return blocking_cols
    
    def is_win(self):
        """Checks if the current player has won."""
        bb = self.occupied_bb ^ self.position_bb  # Current player's bitboard

        # Check vertical
        m = bb & (bb >> 1)
        if m & (m >> 2):
            return True
        # Check horizontal
        m = bb & (bb >> STRIDE)
        if m & (m >> (2 * STRIDE)):
            return True
        # Check diagonal
        m = bb & (bb >> (STRIDE + 1))
        if m & (m >> (2 * (STRIDE + 1))):
            return True
        # Check anti-diagonal
        m = bb & (bb >> (STRIDE - 1))
        if m & (m >> (2 * (STRIDE - 1))):
            return True
        
        return False

    def is_draw(self):
        return self.empt_count == 0
    
    def make_move(self, col):
        """Drops a piece into the specified column."""
        self.position_bb ^= self.occupied_bb
        self.occupied_bb |= 1 << self.heights[col]
        
        self.heights[col] += 1
        self.player_idx ^= 1

        self.empt_count -= 1

    def undo_move(self, col):
        """Removes the top piece from the specified column."""
        self.heights[col] -= 1
        self.player_idx ^= 1
        
        self.occupied_bb ^= 1 << self.heights[col]
        self.position_bb ^= self.occupied_bb

        self.empt_count += 1

    def get_key(self):
        """Returns a unique key for the current board state."""
        return self.position_bb + self.occupied_bb + BOTTOM_MASK

    def get_canonical_key(self):
        """Returns the canonical key for the current board state."""
        board_key = self.get_key()
        symm_key = mirror_key(board_key)
        is_mirrored = board_key > symm_key
        cano_key = symm_key if is_mirrored else board_key

        return cano_key, is_mirrored


if __name__ == "__main__":
    game = Connect4()

    print("Connect 4 - Player 1 = X, Player 2 = O")
    while True:
        print(f"\nCurrent player: {game.player_label()}")
        print(game)

        while True:
            choice = input(f"{game.player_label()}: ").strip()

            # Validate input
            if choice.lower() in ["exit", "quit", "q", "e"]:
                print("Exiting the game.")
                exit()
            if not choice.isdigit():
                print("Please enter a valid number.")
                continue
            col_choice = int(choice)
            if not (0 <= col_choice < COLS):
                print(f"Column out of range. choose between 0 and {COLS - 1}")
                continue
            if not game.is_valid_move(col_choice):
                print("Column full. Choose another one.")
                continue

            break

        game.make_move(col_choice)

        if game.is_win():
            print("\nFinal Board:")
            print(game)
            print(f"{game.player_label(game.player_idx^1)} wins!")
            break
        elif game.is_draw():
            print("\nFinal Board:")
            print(game)
            print("It's a draw!")
            break