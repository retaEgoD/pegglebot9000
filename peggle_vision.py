import numpy as np
import cv2
import easyocr
import logging
from re import sub
from datetime import datetime


# Image templates
BALL_TEMPLATE = cv2.cvtColor(cv2.imread('assets/shooting_ball.png'), cv2.COLOR_RGB2BGR)
BALL_MASK = cv2.imread('assets/shooting_ball_mask.png')
PEG_TEMPLATE = cv2.imread('assets/OrangePeg.png')
PEG_MASK = cv2.imread('assets/Peg_mask.png')
BUCKET_TEMPLATE = cv2.cvtColor(cv2.imread('assets/CatcherA.png'), cv2.COLOR_RGB2BGR)
WIN_TEMPLATE = cv2.cvtColor(cv2.imread('assets/win.png'), cv2.COLOR_RGB2BGR)
FAIL_TEMPLATE = cv2.cvtColor(cv2.imread('assets/fail.png'), cv2.COLOR_RGB2BGR)

DEBUG_SCREENSHOT_FOLDER = 'screenshots'

# HSV colour bounds. Format is lower bound, upper bound.
PEG_BOUNDS = { "orange": (np.array([0, 150, 150]), np.array([5, 255, 255])), 
                "blue": (np.array([226, 83, 42]), np.array([5, 255, 255])),
                "green": (np.array([0, 0, 0]), np.array([0, 0, 0])),
                "purple": (np.array([0, 0, 0]), np.array([0, 0, 0]))
              }

# Template match thresholds
PEG_THRESHOLD = 0.60
BALL_THRESHOLD = 0.65
BUCKET_THRESHOLD = 0.65
GAME_END_THRESHOLD = 0.99

# Histogram comparison threshold
HIST_THRESHOLD = 3

#TODO Add thresholds and detection for all peg colours. Add brick detection. Throw error when shoot check has empty matches.


def convert_mask_to_greyscale(mask):
        """
        Prepares mask for masking by applying threshold and greyscaling.

        Args:
            mask (NumPy Array): Representation of mask as a NumPy array.

        Returns:
            NumPy Array: The prepared mask.
        """
        return cv2.cvtColor(cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)


def create_portrait_mask(board_screenshot):
    board_x = board_screenshot.shape[1]
    mask = np.zeros(board_screenshot.shape[:2], dtype=np.uint8)
    circle_centre = (board_x//2, 20)
    circle_radius = 105
    cv2.circle(mask, center=circle_centre, radius=circle_radius, color=(255, 255, 255), thickness=-1)
    return mask


def check_ready_to_shoot(board_screenshot, debug=False):
    """
    Checks if the game is ready for the player to shoot the ball by 
    checking if the ball is in the ball shooter.

    Returns:
        bool: True if the game is ready to shoot, False otherwise.
    """
    
    # STEP 1: Template match
    board_height, board_width = board_screenshot.shape[:2]
    pw, ph = (board_width//3), (board_height//4)
    py, px = 0, pw # Both x coordinate and portrait width are one third of the game board width.
    portrait = board_screenshot[py:py+ph, px:px+pw] # Slice screenshot to portrait area
    
    # # STEP 1: Template match
    # portrait_mask = create_portrait_mask(board_screenshot)
    # portrait = cv2.bitwise_and(board_screenshot, board_screenshot, mask=portrait_mask)
    
    # # Crop out black
    # greyscale_portrait = cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY)
    # _, portrait_threshold = cv2.threshold(greyscale_portrait, 0, 255, cv2.THRESH_OTSU)
    # bounding_box = cv2.boundingRect(portrait_threshold)
    # x, y, width, height = bounding_box
    # portrait = portrait[y:y+height, x:x+width]
    
    
    matches = cv2.matchTemplate(portrait, BALL_TEMPLATE, cv2.TM_CCOEFF_NORMED, None, mask=convert_mask_to_greyscale(BALL_MASK))
    _, max_val, _, max_loc = cv2.minMaxLoc(matches, None)
    
    
    # STEP 2: Histogram comparison to original image
    
    # Create hsv ball screenshots
    hsv_ball = cv2.cvtColor(BALL_TEMPLATE, cv2.COLOR_BGR2HSV)
    
    ball_x, ball_y = max_loc
    ball_height, ball_width = BALL_TEMPLATE.shape[:2]
    ball_candidate = portrait[ball_y:ball_y+ball_height, ball_x:ball_x+ball_width] # Slice screenshot to just ball
    hsv_ball_candidate = cv2.cvtColor(ball_candidate, cv2.COLOR_BGR2HSV)
    
    # Set histogram settings
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    
    channels = [0, 1] # H and S channels
    
    ball_mask_greyscale = convert_mask_to_greyscale(BALL_MASK) # Greyscale necessary to work as a mask.
    
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
        cv2.rectangle(copy, (ball_x, ball_y), (ball_x + BALL_TEMPLATE.shape[1], ball_y + BALL_TEMPLATE.shape[0]), (0, 255, 0), 2)
            
        cv2.imshow('Ball', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (max_val >= BALL_THRESHOLD) and (hist_comparison >= HIST_THRESHOLD)


def find_pegs(board_screenshot, debug=False):
    
    # Hide portrait to prevent false positives.
    inverted_portrait_mask = cv2.bitwise_not(create_portrait_mask(board_screenshot))
    portrait_masked_board = cv2.bitwise_and(board_screenshot, board_screenshot, mask=inverted_portrait_mask)
    
    # Generate coordinates for pegs
    matches = cv2.matchTemplate(portrait_masked_board, PEG_TEMPLATE, cv2.TM_CCOEFF_NORMED, None, mask=convert_mask_to_greyscale(PEG_MASK))
    ys, xs = np.where(matches >= PEG_THRESHOLD)
    coords = np.column_stack((ys, xs))
    
    
    if debug:
        copy = board_screenshot.copy()
        peg_height, peg_width = PEG_TEMPLATE.shape[:2]
        for y, x in coords:
            cv2.rectangle(copy, (x, y), (x + peg_width, y + peg_height), (0, 255, 0), 2)
            
        cv2.imshow('Pegs', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return coords


def classify_pegs(board_screenshot, locations, debug=False):
    location_mask = np.zeros(board_screenshot.shape[:2], dtype=np.uint8)
    peg_width, peg_height = PEG_MASK.shape[:2]
    for y, x in locations:
        location_mask[y:y+peg_height, x:x+peg_width] = convert_mask_to_greyscale(PEG_MASK)
    
    location_masked_screenshot = cv2.bitwise_and(board_screenshot, board_screenshot, mask=location_mask)
    
    hsv_masked_screenshot = cv2.cvtColor(np.array(location_masked_screenshot), cv2.COLOR_RGB2HSV) # RGB??? BGR??
    orange_lower_bound, orange_upper_bound = PEG_BOUNDS['orange']
    colour_mask = cv2.inRange(hsv_masked_screenshot, orange_lower_bound, orange_upper_bound)
    res = cv2.bitwise_and(board_screenshot, board_screenshot, mask=colour_mask)
    
    if debug:
        cv2.imshow('Location Mask', location_masked_screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow('Pegs colour match', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return res


def get_peg_info(board_screenshot, debug=False):
    peg_locations = find_pegs(board_screenshot, debug=debug)
    masked_pegs = classify_pegs(board_screenshot, peg_locations, debug=debug)
    
    # Peg location logic
    y, x = np.where(np.any(masked_pegs, axis=2)) # Get locations of all pixels that have colour
    locations = list(zip(x, y))
    return locations


def save_peg_screenshots(board_screenshot):
    masked_pegs = find_pegs(board_screenshot)
    current_time = datetime.today().strftime('%Y-%m-%d %H.%M.%S')
    file_prefix = DEBUG_SCREENSHOT_FOLDER + '/' + current_time
    cv2.imwrite(file_prefix + ' Screenshot.png', board_screenshot)
    cv2.imwrite(file_prefix + ' Masked.png', masked_pegs)
    

def get_bucket_position(board_screenshot):
    bucket_height, bucket_width = BUCKET_TEMPLATE.shape[:2]
    result = cv2.matchTemplate(board_screenshot, BUCKET_TEMPLATE, cv2.TM_CCOEFF_NORMED)
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
    # Find template matches
    win_result = cv2.matchTemplate(board_screenshot, WIN_TEMPLATE, cv2.TM_CCOEFF_NORMED,)
    fail_result = cv2.matchTemplate(board_screenshot, FAIL_TEMPLATE, cv2.TM_CCOEFF_NORMED,)
    
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