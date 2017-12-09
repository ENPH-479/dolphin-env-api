from src import mlp_model
import logging
import pickle
import os
import time
import numpy as np
import random
from src import dp_screenshot, helper, mk_downsampler, key2pad, dp_frames, dp_controller

logger = logging.getLogger(__name__)


class MarioKartAgent:
    """ Class implementing a basic Mario Kart AI agent using conditional probability. """
    game_name = "NABE01"

    def __init__(self, pickled_model_path, delay=0.2):
        """ Create a MarioKart Agent instance.

        Args:
            pickled_model_path: Path to the stored state-decision model file.
            screenshot_folder: Name of the folder where the current Dolphin game's screenshots are stored.
        """
        saved_model_file = open(pickled_model_path, 'rb')
        saved_model_obj = pickle.load(saved_model_file)
        self.fitted_model= saved_model_obj.get('model')
        saved_model_file.close()

        self.frame_delay = delay
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots', self.game_name)
        self.key_map = key2pad.KeyPadMap()
        self.key_states = dict()
        self.p = dp_controller.DolphinController("~/.dolphin-emu/Pipes/pipe").__enter__()

    def process_frame(self):
        """ Process a single frame of the current Dolphin game.

        Note:
            Process means read the state of the game and decide what action to take in this context.
            Dolphin will not take screenshots if the game is paused!
        """
        # Advance the Dolphin frame
        #dp_frames.advance('P')

        # Take screenshot of current Dolphin frame
        dp_screenshot.take_screenshot()
        screenshot_file = os.path.join(self.screenshot_dir, 'NABE01-1.png')
        time.sleep(self.frame_delay)
        self.p.press_button(dp_controller.Button.A)
        try:
            # Downsample the screenshot and calculate dictionary key
            ds_image = mk_downsampler.Downsampler(self.game_name, final_dim=15).downsample(screenshot_file)

            state_key = helper.generate_img_key(ds_image)
            state_key = np.array(state_key)
            state_key = state_key.reshape(-1,225)
            print(self.fitted_model)
            y=self.fitted_model.predict(state_key)
            if y==1:
                self.p.set_stick(dp_controller.Stick.MAIN, x=0.3, y=0.5)
            elif y==2:
                self.p.set_stick(dp_controller.Stick.MAIN, x=0.7, y=0.5)
            else:
                self.p.set_stick(dp_controller.Stick.MAIN, x=0.5, y=0.5)

            # Delete used screenshot file
            os.unlink(screenshot_file)
            #self.p.release_button(dp_controller.Button.A)

        except (IOError, FileNotFoundError):
            logger.warning("Screenshot not found, skipping frame.")
