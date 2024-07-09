import pygetwindow
import pyautogui
import time
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

class PeggleHands:
    
    
    def __init__(self, is_nights=False):
        self._set_peg_window(is_nights)
        
        
    def _set_peg_window(self, is_nights):
        """
        Sets the peg window based on the game version.

        Args:
            is_nights (bool): A boolean value indicating whether the game version is Peggle Nights Deluxe.

        Raises:
            IndexError: If no Peggle instance is open.

        """
        try:
            window_name = 'Peggle Deluxe 1.01' if not is_nights else 'Peggle Nights Deluxe 1.0'
            self.peg_window = pygetwindow.getWindowsWithTitle(window_name)[0]
            self.get_board_corner()
            logging.info(f"Window found at {self.peg_window.topleft}.")
        except IndexError:
            print("Error while retrieving window details: No Peggle instance is open.")
        

    def get_peggle_window_corner(self):
        """
        Returns x and y coordinates of the top left corner of the Peggle window relative to
        the screen's top left corner.
        Assumes there is only one window open.

        Args:
            is_nights (bool, optional): Whether or not the Peggle instance is Peggle Nights. 
            Defaults to False.

        Returns:
            (x, y): The coordinates of the Peggle window.
        """
        return self.peg_window.topleft
    
    
    def get_board_corner(self):
        """
        Returns x and y coordinates of the top left corner of the board relative to 
        the screen's top left corner.
        Top left corner is approx 10% across and 13% down respectively.
        """
        x, y = self.get_peggle_window_corner()
        width, height = self.get_window_dimensions()
        return x + int(width*0.1), y + int(height*0.12)
    

    def get_window_dimensions(self):
        """
        Returns the dimensions of the peg window.

        Returns:
            tuple: A tuple containing the width and height of the peg window.
        """
        return self.peg_window.width, self.peg_window.height
    
    
    def get_board_dimensions(self):
        """
        Returns the dimensions of the game board.
        Board size is approx 80% of the game window. Height needs extra 6% to see the ball bucket.

        Returns:
            tuple: A tuple containing the width and height of the game board.
        """
        width, height = self.get_window_dimensions()
        return int(width*0.8), int(height*0.87)
    
    
    def restore_game_window(self):
        """
        Moves window focus to game window.
        """
        self.peg_window.restore()
        time.sleep(0.2)
        self.peg_window.activate()
        time.sleep(0.2)
        

    def get_peggle_screenshot(self):
        """
        Returns a screenshot of the entire Peggle window. 
        
        Args:
            peggle_coordinates (x, y): The coordinates of the Peggle windows.

        Returns:
            list: A list of numpy arrays representing the Peggle screenshots.
        """
        self.restore_game_window()
        
        x, y = self.get_peggle_window_corner()
        width, height = self.get_window_dimensions()
        
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot)
    
    
    def get_board_screenshot(self):
        """
        Returns a screenshot of the game board.

        Returns:
            numpy.ndarray: The screenshot of the game board as a NumPy array.
        """
        self.restore_game_window()

        x, y = self.get_board_corner()
        board_width, board_height = self.get_board_dimensions()

        screenshot = pyautogui.screenshot(region=(x, y, board_width, board_height))
        return np.array(screenshot)
        
    
    def shoot(self, x, y):
        """
        Shoots a peg at the specified coordinates on the game board.
        The coordinates are relative to the game board corner.

        Args:
            x (int): The x-coordinate of the target location.
            y (int): The y-coordinate of the target location.
        """
        logging.info(f"Shooting peg at ({x}, {y}).")
        self.restore_game_window()
        
        board_x, board_y = self.get_board_corner()
                
        pyautogui.moveTo(x + board_x, y + board_y)
        pyautogui.click()