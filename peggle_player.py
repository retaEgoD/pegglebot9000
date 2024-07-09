from peggle_vision import PeggleVision
from peggle_hands import PeggleHands
from peggle_brain import PeggleBrain, PeggleNaive

import time

THRESHOLD = 0.8


# TODO needs to recognise bucket, error handling, read score, ball recognition is poor, exception handling


class PegglePlayer:
    
    def __init__(self) -> None:
        pass
    
    def play_naive(self):
        interfacer = PeggleHands()
        eyes = PeggleVision()
        brain = PeggleNaive()
        
        screenshot = interfacer.get_board_screenshot()
        
        while not eyes.check_game_end(screenshot):
            
            screenshot = interfacer.get_board_screenshot()
            dimensions = interfacer.get_board_dimensions()
                        
            if eyes.check_ready_to_shoot(dimensions, screenshot):
                locations = eyes.get_peg_info(screenshot)
                shot_x, shot_y = brain.select_shot(locations)
                interfacer.shoot(shot_x, shot_y)
            time.sleep(2)
        score = eyes.get_score(screenshot)
        print(score)
        
    def play_nn(self):
        interfacer = PeggleHands()
        eyes = PeggleVision()
        brain = PeggleBrain().to("cuda")
        
        screenshot = interfacer.get_board_screenshot()
        
        while not eyes.check_game_end(screenshot):
            
            screenshot = interfacer.get_board_screenshot()
            dimensions = interfacer.get_board_dimensions()
                        
            if eyes.check_ready_to_shoot(dimensions, screenshot):
                orange_locations = eyes.get_peg_info(screenshot)
                blue_locations = eyes.get_peg_info(screenshot, want_blue=True)
                shot_x, shot_y = brain.select_shot(orange_locations, blue_locations)
                interfacer.shoot(shot_x, shot_y)
            time.sleep(2)
        score = eyes.get_score(screenshot)
        print(score)

        
def main():
    player = PegglePlayer()
    player.play_nn()
    
    

if __name__ == '__main__':
    main()