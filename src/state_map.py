""" This module contains code for maintaining the basic Mario Kart AI agent's state-decision map. """

import logging
import json
import cv2
import pickle
import os
from src import helper

logger = logging.getLogger(__name__)


class state_map():
    """ Class for storing the basic Mario Kart AI agent's state-decision probabilities.

    Attributes:
        screenshot_dir: Directory where the Dolphin emulator saves screenshots.
        keylog_filename: Name of the file where Dolphin keyboard output was logged to.
        output_dir: Directory where the Dolphin keyboard log is stored.
        image_dir: Directory containing the downsampled Dolphin images.
        state_decision_map: Dictionary containing the state to decision probability mapping.
     """
    def __init__(self):
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.keylog_filename = "log.json"
        self.output_dir = helper.get_output_folder()
        self.image_dir = os.path.join(helper.get_output_folder(), "images")
        self.state_decision_map = dict()

    def populate_map(self):
        """ Populate this state map. """
        # Loop over all the output keylog/downsampled image pairs
        num_files = len(os.listdir(self.image_dir))
        count = 0
        # Open and parse the .json keyboard log information
        with open(os.path.join(self.output_dir, self.keylog_filename), 'r') as key_log_file:
            key_log_data = json.load(key_log_file)

            # Keep track of how many times we have seen each state in total
            state_counts = dict()

            # Process the downsampled images one by one
            while count <= num_files:
                count += 1
                # Read the image data
                image_file_name = "{}.png".format(count)
                image_data = cv2.imread(os.path.join(self.image_dir, image_file_name))
                # Generate a tuple from the image so we can hash
                image_serialized = pickle.dumps(image_data, protocol=0)

                # Parse the key log .json dictionaries
                key_map_boolean = key_log_data['data'][count]['presses']
                key_map_numeric = dict()

                # Convert "true" entries to 1 and "false" entries to 0
                for k in key_map_boolean:
                    if key_map_boolean[k] == True:
                        key_map_numeric[k] = 1

                    elif key_map_boolean[k] == False:
                        key_map_numeric[k] = 0

                # Check if the state already exists in the state map
                if image_serialized in self.state_decision_map:

                    for k in self.state_decision_map[image_serialized]:
                        if key_map_numeric[k] == 1:
                            self.state_decision_map[image_serialized][k] +=1

                    state_counts[image_serialized] += 1

                else:
                    self.state_decision_map[image_serialized] = key_map_numeric
                    state_counts[image_serialized] = 1

            # Normalize the entries in the state decision map
            for k in self.state_decision_map:
                for key in self.state_decision_map[k]:
                    self.state_decision_map[k][key] = self.state_decision_map[k][key]/state_counts[k]

