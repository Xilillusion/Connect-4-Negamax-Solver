import math

import pygame
from OpenGL.GL import *
from pygame.locals import *

from solver.connect4_array import Connect4


class Connect4Visual:
    WIDTH = 700
    HEIGHT = 700

    def __init__(self):
        board = [[0] * 7 for _ in range(6)]
        self.game = Connect4(board)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT), DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Connect 4 OpenGL")
        self.font = pygame.font.Font(None, 32)
        self.init_gl()
        self.message = "Player 1's turn (X)"
        self.game_over = False
        self.mouse_col = None
        self.last_move = None

    def init_gl(self):
        glViewport(0, 0, self.WIDTH, self.HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.WIDTH, self.HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def draw_rect(self, x, y, w, h, color):
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def draw_circle(self, x, y, radius, color, segments=60):
        glColor4f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(segments + 1):
            angle = i * 2.0 * 3.14159265 / segments
            glVertex2f(x + radius * math.cos(angle), y + radius * math.sin(angle))
        glEnd()

    def draw_circle_outline(self, x, y, radius, width, color, segments=60):
        glColor4f(*color)
        glLineWidth(width)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = i * 2.0 * 3.14159265 / segments
            glVertex2f(x + radius * math.cos(angle), y + radius * math.sin(angle))
        glEnd()

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        surface = self.font.render(text, True, color)
        data = pygame.image.tostring(surface, "RGBA", True)
        glRasterPos2i(int(x), int(y))
        glDrawPixels(surface.get_width(), surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw_text_with_bg(self, text, x, y, bg_color=(0.1, 0.2, 0.4, 0.8), text_color=(255, 255, 255)):
        surface = self.font.render(text, True, text_color)
        width, height = surface.get_width(), surface.get_height()
        # Draw background rectangle
        self.draw_rect(x - 10, y - 5, width + 20, height + 10, bg_color)
        # Draw text
        data = pygame.image.tostring(surface, "RGBA", True)
        glRasterPos2i(int(x), int(y))
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def reset_game(self):
        self.game = Connect4([[0] * 7 for _ in range(6)])
        self.game_over = False
        self.mouse_col = None
        self.last_move = None
        self.message = "Player 1's turn (X)"

    def update_message(self):
        if self.game_over:
            return
        self.message = f"Player {self.game.player}'s turn ({'X' if self.game.player == 1 else 'O'})"

    def get_board_geometry(self):
        margin = 60
        board_w = self.WIDTH - 2 * margin
        board_h = self.HEIGHT - 2 * margin
        x0 = margin
        y0 = margin
        cell_w = board_w / self.game.cols
        cell_h = board_h / self.game.rows
        return x0, y0, cell_w, cell_h

    def column_at_position(self, x, y):
        x0, y0, cell_w, cell_h = self.get_board_geometry()
        if y < y0 or y > y0 + self.game.rows * cell_h:
            return None
        col = int((x - x0) / cell_w)
        if 0 <= col < self.game.cols:
            return col
        return None

    def handle_move(self, col):
        if self.game_over or col is None:
            return
        if self.game.heights[col] < 0:
            self.message = f"Column {col + 1} is full. Try another column."
            return
        self.game.make_move(col)
        self.last_move = col
        score = self.game.get_score(self.last_move)
        if score == -1:
            winner = 3 - self.game.player
            self.message = f"Player {winner} ({'X' if winner == 1 else 'O'}) wins! Press R to restart."
            self.game_over = True
        elif score == 0:
            self.message = "It's a draw! Press R to restart."
            self.game_over = True
        else:
            self.update_message()

    def draw_board(self):
        x0, y0, cell_w, cell_h = self.get_board_geometry()
        board_w = self.game.cols * cell_w
        board_h = self.game.rows * cell_h
        board_x = x0 - 20
        board_y = y0 - 20
        self.draw_rect(0, 0, self.WIDTH, self.HEIGHT, (0.02, 0.05, 0.12, 1.0))
        self.draw_rect(board_x, board_y, board_w + 40, board_h + 40, (0.04, 0.16, 0.35, 1.0))

        if self.mouse_col is not None and not self.game_over:
            highlight_x = x0 + self.mouse_col * cell_w
            self.draw_rect(highlight_x, y0, cell_w, board_h, (1.0, 1.0, 1.0, 0.08))

        for row in range(self.game.rows):
            for col in range(self.game.cols):
                cx = x0 + col * cell_w + cell_w * 0.5
                cy = y0 + row * cell_h + cell_h * 0.5
                self.draw_circle(cx, cy, min(cell_w, cell_h) * 0.48, (0.07, 0.18, 0.36, 1.0))

                value = int(self.game.board[row][col])
                if value == 0:
                    self.draw_circle(cx, cy, min(cell_w, cell_h) * 0.34, (0.92, 0.92, 0.95, 0.35))
                    self.draw_circle_outline(cx, cy, min(cell_w, cell_h) * 0.34, 2, (0.75, 0.85, 0.95, 0.5))
                else:
                    disc_color = (1.0, 0.38, 0.28, 1.0) if value == 1 else (0.98, 0.94, 0.24, 1.0)
                    shadow_color = (0.0, 0.0, 0.0, 0.18)
                    self.draw_circle(cx, cy + 4, min(cell_w, cell_h) * 0.34, shadow_color)
                    self.draw_circle(cx, cy, min(cell_w, cell_h) * 0.32, disc_color)
                    self.draw_circle_outline(cx, cy, min(cell_w, cell_h) * 0.32, 3, (0.05, 0.05, 0.05, 0.45))

        if self.last_move is not None:
            row = self.game.heights[self.last_move] + 1
            col = self.last_move
            cx = x0 + col * cell_w + cell_w * 0.5
            cy = y0 + row * cell_h + cell_h * 0.5
            self.draw_circle_outline(cx, cy, min(cell_w, cell_h) * 0.38, 5, (1.0, 1.0, 1.0, 0.65))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_r:
                        self.reset_game()
                    elif event.key in [K_1, K_2, K_3, K_4, K_5, K_6, K_7]:
                        col = event.key - K_1
                        self.handle_move(col)
                elif event.type == MOUSEMOTION:
                    self.mouse_col = self.column_at_position(*event.pos)
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    col = self.column_at_position(*event.pos)
                    self.handle_move(col)

            glClearColor(0.02, 0.05, 0.14, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            self.draw_board()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            self.draw_text(self.message, 24, 22)
            self.draw_text("Click a column or press 1-7. Press R to restart.", 24, 54)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    app = Connect4Visual()
    app.run()


if __name__ == "__main__":
    main()
