""" This module contains code for maintaining the basic Mario Kart AI agent's state-decision map. """

import logging
import json

import cv2
import pickle
import os
from src import helper, keylog

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

    def __init__(self, training_keylog="log.json", model_name="naive_model", clean_imgs=False):
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.keylog_filename = training_keylog
        self.output_dir = helper.get_output_folder()
        self.image_dir = os.path.join(helper.get_output_folder(), "images")
        self.models_dir = os.path.join(helper.get_models_folder())
        self.model_name = model_name
        self.clean_imgs = clean_imgs
        self.state_counts = dict()
        self.state_decision_map = dict()
        self.defaults = {k.name: 0 for k in keylog.Keyboard}

    def train(self):
        """ Populate this state map with training data. """

        # Open and parse the .json keyboard log information
        with open(os.path.join(self.output_dir, self.keylog_filename), 'r') as keylog_file:
            key_log_data = json.load(keylog_file).get('data')

            # Loop over all the output keylog/downsampled image pairs
            for state in key_log_data:
                count = state.get('count')
                # Read the image data
                image_file_name = "{}.png".format(count)
                img_path = os.path.join(self.image_dir, image_file_name)
                image_data = cv2.imread(img_path)

                # Generate tuple from flattened image array as dict key.
                key = helper.generate_img_key(image_data)

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

                # Delete downsampled image if True
                if self.clean_imgs:
                    os.unlink(img_path)

            # Laplace Smoothing for default actions
            num_states = sum(self.state_counts.values())
            for value in self.state_decision_map.values():
                for k in value:
                    self.defaults[k] += value[k]
            for k in self.defaults:
                self.defaults[k] = (self.defaults[k] + 1) / (num_states + 2)

            # Convert entries to probabilities in the state decision map
            for k in self.state_decision_map:
                for key_name in self.state_decision_map[k]:
                    self.state_decision_map[k][key_name] /= self.state_counts[k]

        # Update with previously pickled model
        self.state_decision_map, self.defaults, self.state_counts = self.update_prev_model()

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
                state_counts = pfile.get('state_counts')

                # avg model probabilities
                for k, v in self.state_decision_map.items():
                    model[k] = self._avg_key_probs(model[k], v) if model.get(k) else v

                # avg default key press probabilities
                defaults = self._avg_key_probs(defaults, self.defaults)

                # add state counts
                for k in self.state_counts:
                    state_counts[k] = state_counts.setdefault(k, 0) + 1

                return model, defaults, state_counts

        except FileNotFoundError:
            # no previous model found, return current model
            return self.state_decision_map, self.defaults, self.state_counts

    def pickle_model(self):
        """ Save model as a pickled file in models/ """
        model_output = os.path.join(self.models_dir, "{}.pickle".format(self.model_name))
        with open(model_output, 'wb') as m:
            pfile = {"model": self.state_decision_map, "defaults": self.defaults, "state_counts": self.state_counts}
            pickle.dump(pfile, m, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _avg_key_probs(key_set_1, key_set_2):
        new_keys = dict()
        for key in key_set_1:
            new_keys[key] = (key_set_1[key] + key_set_2[key]) / 2.0
        return new_keys
