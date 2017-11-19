""" This module contains code for maintaining the basic Mario Kart AI agent's state-decision map. """

import logging
import json
import cv2
import pickle
import os
from src import helper

logger = logging.getLogger(__name__)


class StateModel:
    """ Class for storing the basic Mario Kart AI agent's state-decision probabilities in a lookup table.

    Attributes:
        screenshot_dir: Directory where the Dolphin emulator saves screenshots.
        keylog_filename: Name of the file where Dolphin keyboard output was logged to.
        output_dir: Directory where the Dolphin keyboard log is stored.
        image_dir: Directory containing the downsampled Dolphin images.
        state_decision_map: Dictionary containing the state to decision probability mapping.
     """

    def __init__(self, keylog_filename="log.json"):
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.keylog_filename = keylog_filename
        self.output_dir = helper.get_output_folder()
        self.image_dir = os.path.join(helper.get_output_folder(), "images")
        self.state_decision_map = dict()

    def populate_map(self):
        """ Populate this state map. """
        # Loop over all the output keylog/downsampled image pairs
        # Open and parse the .json keyboard log information
        with open(os.path.join(self.output_dir, self.keylog_filename), 'r') as keylog_file:
            key_log_data = json.load(keylog_file).get('data')

            # Keep track of how many times we have seen each state in total
            state_counts = dict()
            num_states = 0

            for state in key_log_data:
                count = state.get('count')
                # Read the image data
                image_file_name = "{}.png".format(count)
                image_data = cv2.imread(os.path.join(self.image_dir, image_file_name))
                image_single_channel = image_data[:, :, 1]

                # Generate tuple from flattened image array as dict key.
                key = tuple(image_single_channel.flatten())

                # fetch key presses for current frame from log.json
                key_map_boolean = state.get('presses')
                # Convert "true" entries to 1 and "false" entries to 0
                key_map_numeric = {k: 1 if key_map_boolean[k] else 0 for k in key_map_boolean}

                # Check if the state already exists in the state map
                if key in self.state_decision_map:
                    # increment key press counts for current state
                    for k in key_map_numeric:
                        self.state_decision_map[key][k] += key_map_numeric[k]
                    state_counts[key] += 1
                else:
                    self.state_decision_map[key] = key_map_numeric
                    state_counts[key] = 1

            # Normalize the entries in the state decision map
            for k in self.state_decision_map:
                for key in self.state_decision_map[k]:
                    self.state_decision_map[k][key] = self.state_decision_map[k][key] / state_counts[k]

            # TODO update with previously pickled model
            # TODO pickle the model and save it somewhere
