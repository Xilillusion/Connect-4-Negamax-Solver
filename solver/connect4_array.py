class Connect4:
    def __init__(self, board, connect: int = 4):
        self.board = board
        self.rows, self.cols = len(board), len(board[0])
        self.connect = connect
        
        self.heights = [self.rows-1] * self.cols
        self.player = 1

        # Initialize heights from the given board state
        for r in range(self.rows):
            for c in range(self.cols):
                player = board[r][c]
                if player != 0:
                    self.heights[c] -= 1
                    self.player = 3 - self.player

    def __str__(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        res = []
        for row in self.board:
            res.append(' '.join(symbols[cell] for cell in row))
        res.append("-" * (self.cols * 2 - 1))
        res.append(' '.join(str(i + 1) for i in range(self.cols)))
        return "\n".join(res)
    
    def check_line(self, y, x, player):
        def count_dir(dy, dx):
            count = 0
            r, c = y + dy, x + dx
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                count += 1
                r += dy
                c += dx
            return count
        
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            if 1 + count_dir(dy,dx) + count_dir(-dy,-dx) >= self.connect:
                return True
        return False
    
    def get_score(self, x):
        """
        Evaluate the board state after the last move for a win/loss/draw.
        Returns -1 if opponent wins, 0 for draw, None if ongoing.
        """
        if x is None:
            return None
        
        y = self.heights[x] + 1
        last_player = 3 - self.player

        if self.check_line(y, x, last_player):
            return -1
            
        if sum(self.heights) == -self.cols:
            return 0
        return None
    
    def make_move(self, x):
        """Drop a piece into column x."""
        self.board[self.heights[x]][x] = self.player
        
        self.heights[x] -= 1
        self.player = 3 - self.player

    def undo_move(self, x):
        """Undo a piece in column x."""
        self.heights[x] += 1
        self.player = 3 - self.player

        self.board[self.heights[x]][x] = 0

if __name__ == "__main__":
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    connect = 4
    game = Connect4(board, connect)

    def current_player_text():
        return f"Player {game.player} ({'X' if game.player == 1 else 'O'})"

    print("Connect 4 - Player 1 = X, Player 2 = O")
    while True:
        print(f"Current player: {current_player_text()}")
        print(game)

        col = None
        while col is None:
            choice = input(f"{current_player_text()}, choose a column 1-{game.cols}: ")
            if not choice.isdigit():
                print("Please enter a number.")
                continue
            col_choice = int(choice) - 1
            if col_choice < 0 or col_choice >= game.cols:
                print("That column is out of range.")
                continue
            if game.heights[col_choice] < 0:
                print("That column is full.")
                continue
            col = col_choice

        game.make_move(col)
        score = game.get_score(col)

        if score == -1:
            print(game)
            winner = 3 - game.player
            print(f"Player {winner} ({'X' if winner == 1 else 'O'}) wins!")
            break
        elif score == 0:
            print(game)
            print("It's a draw!")
            break
