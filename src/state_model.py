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

    def __init__(self, training_keylog="log.json", model_name="naive_model"):
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.keylog_filename = training_keylog
        self.output_dir = helper.get_output_folder()
        self.image_dir = os.path.join(helper.get_output_folder(), "images")
        self.models_dir = os.path.join(helper.get_models_folder())
        self.model_name = model_name
        self.state_counts = dict()
        self.state_decision_map = dict()
        self.defaults = dict()

    def populate_map(self):
        """ Populate this state map. """
        # Loop over all the output keylog/downsampled image pairs
        # Open and parse the .json keyboard log information
        with open(os.path.join(self.output_dir, self.keylog_filename), 'r') as keylog_file:
            key_log_data = json.load(keylog_file).get('data')

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
                else:
                    self.state_decision_map[key] = key_map_numeric

                # Keep track of how many times we have seen each state in total
                self.state_counts[key] = self.state_counts.setdefault(key, 0) + 1

            # Convert entries to probabilities in the state decision map
            for k in self.state_decision_map:
                for key_name in self.state_decision_map[k]:
                    self.state_decision_map[k][key_name] /= self.state_counts[k]

        # Update with previously pickled model
        self.state_decision_map, self.defaults = self.update_prev_model()

        # Pickle the model and save it in models/
        self.pickle_model()

    def update_prev_model(self):
        """ Updates current trained model with previously saved model """
        model_file = os.path.join(self.models_dir, "{}.pickle".format(self.model_name))
        try:
            # load model
            with open(model_file, 'rb') as mf:
                pfile = pickle.load(mf)
                model = pfile.get('model')
                defaults = pfile.get('defaults')

                # avg model probabilities
                for k, v in self.state_decision_map.items():
                    model[k] = self._avg_key_probs(model[k], v) if model.get(k) else v

                # avg default key press probabilities
                defaults = self._avg_key_probs(defaults, self.defaults)

                return model, defaults

        except FileNotFoundError:
            # no previous model found, return current model
            return self.state_decision_map, self.defaults

    def pickle_model(self):
        model_output = os.path.join(self.models_dir, "{}.pickle".format(self.model_name))
        with open(model_output, 'wb') as m:
            pfile = {"model": self.state_decision_map, "defaults": self.defaults}
            pickle.dump(pfile, m, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _avg_key_probs(key_set_1, key_set_2):
        new_keys = dict()
        for key in key_set_1:
            new_keys[key] = (key_set_1[key] + key_set_2[key]) / 2.0
        return new_keys
