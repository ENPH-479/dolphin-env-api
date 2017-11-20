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
import time
import random
from src import dp_screenshot, helper, mk_downsampler, key2pad, dp_frames

class MarioKartAgent:
    """ Class implementing a basic Mario Kart AI agent using conditional probability. """

    def __init__(self, pickled_model_path, screenshot_folder):
        """ Create a MarioKart Agent instance.

        Args:
            pickled_model_path: Path to the stored state-decision model file.
            screenshot_folder: Name of the folder where the current Dolphin game's screenshots are stored.
        """
        saved_model_file = open(pickled_model_path, 'rb')
        saved_model_obj = pickle.load(saved_model_file)
        self.decision_map = saved_model_obj.get("model")
        self.default_map = saved_model_obj.get("defaults")
        saved_model_file.close()
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.screenshot_folder = screenshot_folder
        self.key_map = key2pad.KeyPadMap()
        self.key_states = dict()

    def process_frame(self):
        """ Process a single frame of the current Dolphin game.

        Note:
            Process means read the state of the game and decide what action to take in this context.
            Dolphin will not take screenshots if the game is paused!
        """
        # Advance the Dolphin frame
        dp_frames.advance('P')

        # Take screenshot of current Dolphin frame
        dp_screenshot.take_screenshot()
        screenshot_path = os.path.join(self.screenshot_dir, self.screenshot_folder)
        screenshot_file = os.path.join(screenshot_path, 'NABE01-1.png')
        time.sleep(0.3)

        if os.path.isfile(screenshot_file):
            # Downsample the screenshot and calculate dictionary key
            ds_image = mk_downsampler.Downsampler('NABE01', final_dim = 15).downsample(screenshot_file)
            state_key = tuple(ds_image.flatten())

            # Look up the game state to decide which action to take.
            if state_key in self.decision_map:
                # Choose which action to take using the key press probabilities in the decision map
                for key_name in self.decision_map[state_key]:
                    rand_num = random.uniform(0,1)
                    if rand_num > self.decision_map[state_key][key_name]:
                        self.key_states[key_name] = False
                    else:
                        self.key_states[key_name] = True
            else:
                # Choose which action to take using the key press probabilities in the default map
                for key_name in self.default_map:
                    rand_num = random.uniform(0,1)
                    if rand_num > self.default_map[key_name]:
                        self.key_states[key_name] = False
                    else:
                        self.key_states[key_name] = True

            # Send the updated key states to the Dolphin controller
            self.key_map.update(self.key_states)

            # Cleanup the Dolphin screenshot folder
            os.remove(screenshot_file)

