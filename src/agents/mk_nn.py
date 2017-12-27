"""
This module implements a Mario Kart AI agent using a Neural Network.
"""

import logging
import os
import time

import torch

from src import dp_screenshot, helper, mk_downsampler, key2pad
from src.agents.mk_nn_train import MKRNN

logger = logging.getLogger(__name__)


class MarioKartNN:
    """ Class implementing a neural network Mario Kart AI agent """
    game_name = "NABE01"

    def __init__(self, pickled_model_path, delay=0.2):
        """ Create a MarioKart Agent instance.
        Args:
            pickled_model_path: Path to the neural network model file.
        """
        self.model = torch.load(pickled_model_path)

        self.frame_delay = delay
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots', self.game_name)
        self.key_map = key2pad.KeyPadMap()

    def process_frame(self):
        """ Process a single frame of the current Dolphin game.
        Note:
            Process means read the state of the game and decide what action to take in this context.
            Dolphin will not take screenshots if the game is paused!
        """
        # Advance the Dolphin frame
        # dp_frames.advance('P')

        # Take screenshot of current Dolphin frame
        dp_screenshot.take_screenshot()
        screenshot_file = os.path.join(self.screenshot_dir, 'NABE01-1.png')
        time.sleep(self.frame_delay)

        try:
            # Downsample the screenshot and calculate dictionary key
            ds_image = mk_downsampler.Downsampler(self.game_name, final_dim=15).downsample(screenshot_file)

            # Generate tensor from image
            x = helper.get_tensor(ds_image)

            if x is None:
                logger.info("Skipping frame - no input received.")
                return

            # Predict key presses using neural network

            #For mkcnn only
            x=x[None,:,:]
            x=torch.stack((x,x,x,x,x))
            x=x.view(1,5,15,15)
            x=x.float()

            prediction = self.model(x)
            
            #For mkcnn only
            prediction = prediction[0,5,1,:]
            prediction= prediction[None,:]

            # Choose which action to take from prediction
            key_state = helper.get_key_state_from_vector(prediction)

            # Send updated key states to the Dolphin controller
            self.key_map.update(key_state)

            # Delete used screenshot file
            os.unlink(screenshot_file)

        except (IOError, FileNotFoundError):
            logger.warning("Screenshot not found, skipping frame.")