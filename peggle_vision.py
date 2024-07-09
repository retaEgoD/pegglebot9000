import numpy as np
import cv2
import easyocr
import logging

BALL_TEMPLATE = cv2.cvtColor(cv2.imread('assets/shooting_ball.png'), cv2.COLOR_RGB2BGR)
WIN_TEMPLATE = cv2.cvtColor(cv2.imread('assets/win.png'), cv2.COLOR_RGB2BGR)
FAIL_TEMPLATE = cv2.cvtColor(cv2.imread('assets/fail.png'), cv2.COLOR_RGB2BGR)

ORANGE_LOWER_BOUND = np.array([0, 150, 150])
ORANGE_UPPER_BOUND = np.array([5, 255, 255])
ORANGE_BOUNDS = (ORANGE_LOWER_BOUND, ORANGE_UPPER_BOUND)

BLUE_LOWER_BOUND = np.array([226, 83, 42])
BLUE_UPPER_BOUND = np.array([222, 56, 74])
BLUE_BOUNDS = (BLUE_LOWER_BOUND, BLUE_UPPER_BOUND)

BALL_THRESHOLD = 0.65
GAME_END_THRESHOLD = 0.99


class PeggleVision:
    
    def __init__(self):
        pass
    
    # TODO: Adjust so that it doesn't shoot if "Balls left" is on the screen.
    def check_ready_to_shoot(self, board_dimensions, board_screenshot, debug=False):
        """
        Checks if the game is ready for the player to shoot the ball by 
        checking if the ball is in the ball shooter.

        Returns:
            bool: True if the game is ready to shoot, False otherwise.
        """
        
        # TODO: Combine with color matching to be more robust. Reduce threshold as well.
        board_width, board_height = board_dimensions
        portrait_width, portrait_height = board_width//3, board_height//4
        y, x = 0, portrait_width # Both x and portrait width are one third of the game board width.
        
        # Slice screenshot to portrait area, convert to greyscale and load ball template
        board_screenshot = board_screenshot[y:y+portrait_height, x:x+portrait_width]
        
        # Match template
        result = cv2.matchTemplate(board_screenshot, BALL_TEMPLATE, cv2.TM_CCOEFF_NORMED)
        coords = np.where(result >= BALL_THRESHOLD)
        
        # Display portrait screenshot with ball location if debugging.
        if debug:
            copy = board_screenshot.copy()
            for pt in zip(*coords[::-1]):
                cv2.rectangle(copy, pt, (pt[0] + BALL_TEMPLATE.shape[1], pt[1] + BALL_TEMPLATE.shape[0]), (0, 255, 0), 2)
            cv2.imshow('Ball', copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.any(coords)


    def get_peg_info(self, board_screenshot, want_blue=False):
        # TODO: Consider dropping hsv in favour of direct colour match. Also consider combining with template matching. Blue matching does not work.
        hsv_screenshot = cv2.cvtColor(np.array(board_screenshot), cv2.COLOR_RGB2HSV)
        bounds = ORANGE_BOUNDS if not want_blue else BLUE_BOUNDS
            
        mask = cv2.inRange(hsv_screenshot, bounds[0], bounds[1])
        res = cv2.bitwise_and(board_screenshot, board_screenshot, mask=mask)
        
        # Peg location logic
        y, x = np.where(np.any(res, axis=2)) # Get locations of all pixels that have colour
        locations = list(zip(x, y))
        return locations
        
    
    def get_bucket_position(self, board_screenshot):
        pass
    
    
    def check_game_end(self, board_screenshot):
        """Check if the game has finished by checking for the win or fail screen.
        

        Args:
            board_screenshot (NumPy Array): A screenshot of the game board.

        Returns:
            boolean: True if the game has finished, False otherwise.
        """
        # Find template matches
        win_result = cv2.matchTemplate(board_screenshot, WIN_TEMPLATE, cv2.TM_CCOEFF_NORMED,)
        fail_result = cv2.matchTemplate(board_screenshot, FAIL_TEMPLATE, cv2.TM_CCOEFF_NORMED,)
        
        # Get locations of matches above threshold
        win_loc = np.where(win_result >= GAME_END_THRESHOLD)
        fail_loc = np.where(fail_result >= GAME_END_THRESHOLD)
        
        # Game is finished if match is found
        if (np.logical_or(any(win_loc), any(fail_loc))):
            logging.info("Game has finished")
            return True
        return False
    
    
    def get_score(self, board_screenshot):
        reader = easyocr.Reader(['en'], gpu=True)
        text = reader.readtext(board_screenshot)
        score_str = text[2][1] # Score is always the third elemtn
        score = int(score_str.replace(',', ''))
        return score