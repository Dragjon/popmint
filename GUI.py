import pygame
import time

def print_board(board, display_time=2):
    # Constants
    WINDOW_SIZE = 600
    GRID_SIZE = len(board)
    CELL_SIZE = WINDOW_SIZE // GRID_SIZE
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHT_GREEN = (144, 238, 144)
    ORANGE = (255, 165, 0)

    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Pop-It Board")

    def draw_board():
        screen.fill(WHITE)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                color = LIGHT_GREEN if board[r][c] == 0 else ORANGE
                pygame.draw.rect(screen, color, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLACK, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # Draw the board
    draw_board()
    pygame.display.flip()