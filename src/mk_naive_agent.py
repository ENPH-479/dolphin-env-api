""" This module implements a basic Mario Kart AI agent using a saved state-decision map.

The basic Mario Kart AI agent functions as follows:
    1) The state-decision map contains the probabilities of taking an action in a certain state (downsampled image)
    2) Using the state-decision map, the basic Mario Kart AI chooses an action to take by drawing a uniformly
        distributed random number between 0 and 1 and comparing it to the probability of pressing a button in a given
        state. If the random number is above the probability threshold, the action is taken. If a state has never been
        encountered before, an action is chosen based on how often that action is taken in total out of all encountered
        states.
    3) The basic Mario Kart AI agent takes the desired action by sending controller inputs to the Dolphin emulator using
        fifo pipes and advancing the game by 1 frame, at which point the process starts again based on the next frame
        (state)
"""

import pickle
import os
import random
from src import dp_screenshot, helper, mk_downsampler, key2pad

class MarioKartAgent:
    """ Create a MarioKart Agent instance.

     Args:
         pickled_model_path: Path to the stored state-decision model file
     """
    def __init__(self, pickled_model_path):
        saved_model_file = open(pickled_model_path, 'rb')
        saved_model_obj = pickle.load(saved_model_file)
        self.decision_map = saved_model_obj.get("model")
        self.default_map = saved_model_obj.get("defaults")
        saved_model_file.close()
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.screenshot_folder = 'NABE01'


    def run(self):
        # TODO complete
        # take screenshot
        screen.get_screen()
        # load img
        screen_dir = os.path.join(self.screenshot_dir, self.game_name)

        # downsample
        downsampling.Downsampler('NABE01', final_dim=15).downsample_dir(save_imgs=True)
        # look up game state

        # generate random number [0,1]
        rand=random.uniform(0,1)
        # take diff of key press state - update with current pressed keys
        # update controller inputs through pipe
        # delete screenshot?
        os.remove(os.path.join(self.screenshot_dir, self.game_name))
        pass
