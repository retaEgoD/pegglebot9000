import peggle_vision as pv
from peggle_hands import PeggleHands
from peggle_brain import PeggleBrain, PeggleNaive

import time
import logging
import keyboard

THRESHOLD = 0.8


# TODO needs to recognise bucket, error handling, read score, ball recognition is poor, exception handling


class PegglePlayer:
    
    def __init__(self) -> None:
        pass
    
    def play_naive(self, autoplay=True, debug=False):
        interfacer = PeggleHands()
        brain = PeggleNaive()
        
        
        while autoplay:
            screenshot = interfacer.get_board_screenshot()
            print("Starting new level...")
            
            while not pv.check_game_end(screenshot):
                
                screenshot = interfacer.get_board_screenshot()
                            
                if pv.check_ready_to_shoot(screenshot, debug=debug):
                    try:
                        locations = pv.get_peg_info(screenshot, debug=debug)
                        shot_x, shot_y = brain.select_shot(locations)
                        interfacer.click(shot_x, shot_y)
                    except IndexError:
                        logging.warning("No orange pegs found. Saving board data. Retrying...")
                        pv.save_peg_screenshots(screenshot)
                        
                if keyboard.is_pressed('esc'):
                    logging.info('ESC pressed. Exiting...')
                    return
                
                if keyboard.is_pressed('s'):
                    pv.save_peg_screenshots(screenshot)
                    logging.info('Screenshot!')
                        
                time.sleep(2)

            score = pv.get_score(screenshot)
            print(f"Score: {score}")
            time.sleep(2)
            
            interfacer.click(315, 370) # TODO SHIT CODE CLEAN UP
    
    
    
    def play_nn(self):
        interfacer = PeggleHands()
        brain = PeggleBrain().to("cuda")
        
        screenshot = interfacer.get_board_screenshot()
        
        while not pv.check_game_end(screenshot):
            
            screenshot = interfacer.get_board_screenshot()
            dimensions = interfacer.get_board_dimensions()
                        
            if pv.check_ready_to_shoot(dimensions, screenshot):
                orange_locations = pv.get_peg_info(screenshot)
                blue_locations = pv.get_peg_info(screenshot, want_blue=True)
                shot_x, shot_y = brain.select_shot(orange_locations, blue_locations)
                interfacer.click(shot_x, shot_y)
            time.sleep(2)
        score = pv.get_score(screenshot)
        print(score)

        
def main():
    player = PegglePlayer()
    # player.play_naive(debug=True)
    player.play_naive()
    
    

if __name__ == '__main__':
    main()