import cv2
import numpy as np
from lib.InvariantTM import invariantMatchTemplate

COLOURS = ['orange', 'green', 'purple']
ROUND_PEGS = [f'assets/{colour}_peg.jpg' for colour in COLOURS]
BRICKS = [f'assets/{colour}_brick.png' for colour in COLOURS]

PEG_IMAGES = [cv2.imread(brick) for brick in BRICKS]

BOARD = cv2.imread('assets/board_test.jpg')
BOARD_BRICKS = cv2.imread('assets/board_bricks.jpg')
THRESHOLD = 0.435
THRESHOLD_BRICK = 0.8


def locate_peg_on_board(board_image, template_image):
    th, tw = template_image.shape[:2]
    board_copy = board_image.copy()

    result = cv2.matchTemplate(board_image, template_image, cv2.TM_SQDIFF_NORMED)

    y_coords, x_coords = np.where(result <= THRESHOLD)
    for x, y, in zip(x_coords, y_coords):
        cv2.rectangle(board_copy, (x, y), (x+tw, y+th), (255, 255, 255), 2)

    cv2.imshow('Board', board_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def locate_brick_on_board(board_image, template_image):
    th, tw = template_image.shape[:2]
    board_copy = board_image.copy()

    result = invariantMatchTemplate(board_image, template_image, "TM_SQDIFF_NORMED", THRESHOLD_BRICK, 0.8, (0, 360), 5, (75, 150), 25, False, False)
    result = cv2.matchTemplate(board_image, template_image, cv2.TM_SQDIFF_NORMED)

    y_coords, x_coords = np.where(result <= THRESHOLD)
    for x, y, in zip(x_coords, y_coords):
        cv2.rectangle(board_copy, (x, y), (x+tw, y+th), (255, 255, 255), 2)

    cv2.imshow('Board', board_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for peg in PEG_IMAGES:
    locate_brick_on_board(BOARD_BRICKS, peg)
