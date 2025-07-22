import numpy as np
import cv2
import easyocr
import logging
from re import sub
from datetime import datetime
import os


# Image templates
BALL_TEMPLATE_PATH = "assets/shooting_ball.png"
BALL_MASK_PATH = 'assets/shooting_ball_mask.png'
PEG_TEMPLATE_PATH = 'assets/orange_peg.png'
PEG_MASK_PATH = 'assets/peg_mask.png'
BRICK_TEMPLATE_PATH = 'assets/orange_brick.png'
BUCKET_TEMPLATE_PATH = 'assets/CatcherA.png'
WIN_TEMPLATE_PATH = 'assets/win.png'
FAIL_TEMPLATE_PATH = 'assets/fail.png'

DEBUG_SCREENSHOT_FOLDER = 'screenshots'

# HSV colour bounds. Format is lower bound, upper bound.
PEG_BOUNDS = { "orange": (np.array([0, 50, 50]), np.array([15, 255, 255])), 
                "blue": (np.array([103, 50, 50]), np.array([123, 255, 255])),
                "green": (np.array([50, 50, 50]), np.array([70, 255, 255])),
                "purple": (np.array([144, 50, 50]), np.array([164, 255, 255]))
              }

# Template match thresholds
PEG_THRESHOLD = 0.55
BRICK_THRESHOLD = 0.3
BALL_THRESHOLD = 0.65
BUCKET_THRESHOLD = 0.65
GAME_END_THRESHOLD = 0.99

# Histogram comparison threshold
HIST_THRESHOLD = 3

#TODO Add brick detection. Throw error when shoot check has empty matches. Update board screenshot function.
# Vision is finding it difficult to detect pegs near wall. Orange borders are also interfering with the orange detection.


def load_image(path):
    """
    Loads an image from an asset path.

    Args:
        path (String): The relative path to the asset.

    Returns:
        NumPy array: The image as a NumPy array.
    """
    abs_path = os.path.join(os.path.dirname(__file__), path) 
    img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_RGB2BGR)
    return img


def convert_mask_to_greyscale(mask):
    """
    Prepares mask for masking by applying threshold and greyscaling.

    Args:
        mask (NumPy Array): Representation of mask as a NumPy array.

    Returns:
        NumPy Array: The prepared mask.
    """
    return cv2.cvtColor(cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)


def create_portrait_masked_board(board_screenshot):
    """
    Creates a mask to cover the portrait area of the game board.

    Args:
        board_screenshot (NumPy array): NumPy array of a screenshot of the game board.

    Returns:
        NumPy Array: A mask covering the portrait area of the board.
    """
    board_x = board_screenshot.shape[1]
    mask = np.ones(board_screenshot.shape[:2], dtype=np.uint8)
    circle_centre = (board_x//2, 20)
    circle_radius = 105
    cv2.circle(mask, center=circle_centre, radius=circle_radius, color=(0, 0, 0), thickness=-1)
    masked_board = cv2.bitwise_and(board_screenshot, board_screenshot, mask=mask)
    return masked_board


def check_ready_to_shoot(board_screenshot, debug=False):
    """
    Checks if the game is ready for the player to shoot the ball by 
    checking if the ball is in the ball shooter.

    Returns:
        bool: True if the game is ready to shoot, False otherwise.
    """
    
    ball_template = load_image(BALL_TEMPLATE_PATH)
    ball_mask = load_image(BALL_MASK_PATH)
    
    # STEP 1: Template match
    
    # Slice screenshot to portrait area    
    board_height, board_width = board_screenshot.shape[:2]
    pw, ph = (board_width//3), (board_height//4)
    py, px = 0, pw # Both x coordinate and portrait width are one third of the game board width.
    portrait = board_screenshot[py:py+ph, px:px+pw]
    
    matches = cv2.matchTemplate(portrait, ball_template, cv2.TM_CCOEFF_NORMED, None, mask=convert_mask_to_greyscale(ball_mask))
    _, max_val, _, max_loc = cv2.minMaxLoc(matches, None)
    
    
    # STEP 2: Histogram comparison to original image
    
    
    # Slice screenshot to just ball candidate
    ball_x, ball_y = max_loc
    ball_height, ball_width = ball_template.shape[:2]
    ball_candidate = portrait[ball_y:ball_y+ball_height, ball_x:ball_x+ball_width] 
    
    # Set histogram settings
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    
    channels = [0, 1] # H and S channels
    
    
    hsv_ball = cv2.cvtColor(ball_template, cv2.COLOR_BGR2HSV)
    hsv_ball_candidate = cv2.cvtColor(ball_candidate, cv2.COLOR_BGR2HSV)
    ball_mask_greyscale = convert_mask_to_greyscale(ball_mask) # Greyscale necessary to work as a mask.
    
    # Calculate histograms and compare
    hist_ball = cv2.calcHist([hsv_ball], channels, ball_mask_greyscale, hist_size, ranges, accumulate=False)
    hist_ball_candidate = cv2.calcHist([hsv_ball_candidate], channels, ball_mask_greyscale, hist_size, ranges, accumulate=False)
    cv2.normalize(hist_ball, hist_ball, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_ball_candidate, hist_ball_candidate, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    
    hist_comparison = cv2.compareHist(hist_ball, hist_ball_candidate, method=cv2.HISTCMP_INTERSECT)
    
    
    # Display portrait screenshot with ball location if debugging. Also log template and histogram info.
    logging.info(f"Max ball value coordinates: {ball_x, ball_y}") 
    logging.info(f"Maximum value: {max_val}")
    logging.info(f"Hist Comparison: {hist_comparison}")
    print()
    if debug:
        
        copy = portrait.copy()
        cv2.rectangle(copy, (ball_x, ball_y), (ball_x + ball_template.shape[1], ball_y + ball_template.shape[0]), (0, 255, 0), 2)
            
        cv2.imshow('Ball', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (max_val >= BALL_THRESHOLD) and (hist_comparison >= HIST_THRESHOLD)


def find_pegs(board_screenshot, bricks, debug=False):
    """
    Locates the coordinates of all pegs on the board using template matching.

    Args:
        board_screenshot (NumPy array): NumPy array of a screenshot of the game board.
        debug (bool, optional): Whether or not to use debug mode. Defaults to False.

    Returns:
        List: x, y coordinates of all pegs on the baord.
    """
    
    brick_template = load_image(BRICK_TEMPLATE_PATH)
    peg_template = load_image(PEG_TEMPLATE_PATH)
    peg_mask = load_image(PEG_MASK_PATH)
    
    # Generate coordinates for pegs
    if bricks:
        matches = cv2.matchTemplate(board_screenshot, brick_template, cv2.TM_CCOEFF_NORMED, None, mask=None)
        ys, xs = np.where(matches >= BRICK_THRESHOLD)
    else:
        matches = cv2.matchTemplate(board_screenshot, peg_template, cv2.TM_CCOEFF_NORMED, None, mask=convert_mask_to_greyscale(peg_mask))
        ys, xs = np.where(matches >= PEG_THRESHOLD)
    coords = np.column_stack((ys, xs))
    
    
    if debug:
        copy = board_screenshot.copy()
        peg_height, peg_width = peg_template.shape[:2]
        for y, x in coords:
            cv2.rectangle(copy, (x, y), (x + peg_width, y + peg_height), (0, 255, 0), 2)
            
        cv2.imshow('Pegs', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return coords


def create_masked_screenshot(board_screenshot, locations, debug=False):
    peg_mask = load_image(PEG_MASK_PATH)
    location_mask = np.zeros(board_screenshot.shape[:2], dtype=np.uint8)
    peg_width, peg_height = peg_mask.shape[:2]
    for y, x in locations:
        location_mask[y:y+peg_height, x:x+peg_width] = convert_mask_to_greyscale(peg_mask)
    location_masked_screenshot = cv2.bitwise_and(board_screenshot, board_screenshot, mask=location_mask)
    
    if debug:
        cv2.imshow('Location Mask', location_masked_screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return location_masked_screenshot


def classify_pegs(board_screenshot, location_masked_screenshot, debug=False):
    
    hsv_masked_screenshot = cv2.cvtColor(np.array(location_masked_screenshot), cv2.COLOR_RGB2HSV) # Not BGR??? Why does this work correctly?
    colour_matched_screenshots = {}
    for colour, (lower_bound, upper_bound) in PEG_BOUNDS.items():
        colour_mask = cv2.inRange(hsv_masked_screenshot, lower_bound, upper_bound)
        res = cv2.bitwise_and(board_screenshot, board_screenshot, mask=colour_mask)
        colour_matched_screenshots[colour] = res
    
    if debug:
        for colour, res in colour_matched_screenshots.items():
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            cv2.imshow(f'{colour} colour match', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return colour_matched_screenshots


def get_peg_info(board_screenshot, debug=False):
    # Hide portrait to prevent false positives.
    portrait_masked_board = create_portrait_masked_board(board_screenshot)
    # brick_coords = find_pegs(portrait_masked_board, True, debug=debug)
    peg_coords = find_pegs(portrait_masked_board, False, debug=debug)
    location_mask = create_masked_screenshot(portrait_masked_board, peg_coords, debug=debug)
    masked_pegs = classify_pegs(portrait_masked_board, location_mask, debug=debug)
    
    # Peg location logic
    y, x = np.where(np.any(masked_pegs['orange'], axis=2)) # Get locations of all pixels that have colour
    locations = list(zip(x, y))
    return locations


def save_peg_screenshots(board_screenshot):
    """
    Saves a screenshot of the current board state and a peg location masked board state.

    Args:
        board_screenshot (NumPy array): NumPy array of a screenshot of the game board.
    """
    portrait_masked_board = create_portrait_masked_board(board_screenshot)
    peg_coords = find_pegs(portrait_masked_board)
    location_mask = create_masked_screenshot(portrait_masked_board, peg_coords)
    
    current_time = datetime.today().strftime('%Y-%m-%d %H.%M.%S')
    file_prefix = DEBUG_SCREENSHOT_FOLDER + '/' + current_time
    cv2.imwrite(file_prefix + ' Screenshot.png', board_screenshot)
    cv2.imwrite(file_prefix + ' Masked.png', location_mask)
    

def get_bucket_position(board_screenshot):
    bucket_template = load_image(BUCKET_TEMPLATE_PATH)
    
    bucket_height, bucket_width = bucket_template.shape[:2]
    result = cv2.matchTemplate(board_screenshot, bucket_template, cv2.TM_CCOEFF_NORMED)
    coords = np.where(result >= BUCKET_THRESHOLD)
    
    # Select first coordinate and get centre of bucket, precision isn't important.
    return (coords[0][0] + bucket_height//2, coords[1][0] + bucket_width//2)


def check_game_end(board_screenshot):
    """Check if the game has finished by checking for the win or fail screen.
    

    Args:
        board_screenshot (NumPy Array): A screenshot of the game board.

    Returns:
        boolean: True if the game has finished, False otherwise.
    """
    win_template = load_image(WIN_TEMPLATE_PATH)
    fail_template = load_image(FAIL_TEMPLATE_PATH)
    
    # Find template matches
    win_result = cv2.matchTemplate(board_screenshot, win_template, cv2.TM_CCOEFF_NORMED,)
    fail_result = cv2.matchTemplate(board_screenshot, fail_template, cv2.TM_CCOEFF_NORMED,)
    
    # Check if game is finished.
    win_loc = np.where(win_result >= GAME_END_THRESHOLD)
    fail_loc = np.where(fail_result >= GAME_END_THRESHOLD)
    
    if (len(win_loc[0]) > 0) or (len(fail_loc[0]) > 0):
        logging.info("Game has finished")
        return True
    return False


def get_score(board_screenshot):
    reader = easyocr.Reader(['en'], gpu=True)
    text = reader.readtext(board_screenshot)
    score_str = text[2][1] # Score is always the third element
    score = int(sub("[,.]", "", score_str))
    return score