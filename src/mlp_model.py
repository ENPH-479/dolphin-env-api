from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import logging
import json

import cv2
import pickle
import os
from src import helper, keylog

from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)

class MlpModel:

    def __init__(self, training_keylog="log.json", model_name="mlp_model", clean_imgs=False):
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
        self.trained_model=1

    def train(self):
        try:
            # Open and parse the .json keyboard log information
            with open(os.path.join(self.output_dir, self.keylog_filename), 'r') as keylog_file:
                key_log_data = json.load(keylog_file).get('data')
                y= np.zeros(1,)
                X= np.zeros(1,)
                #xg_model= XGBClassifier(n_estimators=200, max_depth=4, objective="binary:logistic", learning_rate=0.016,)
                nn_model= MLPClassifier(hidden_layer_sizes=(200,),activation="relu",alpha=0.1,learning_rate='invscaling')
                # Loop over all the output keylog/downsampled image pairs
                for state in key_log_data:
                    count = state.get('count')
                    # Read the image data
                    image_file_name = "{}.png".format(count)
                    img_path = os.path.join(self.image_dir, image_file_name)
                    image_data = cv2.imread(img_path)
                    # Generate tuple from flattened image array as dict key.
                    key = helper.generate_img_key(image_data)

                    if count==1:
                        X=key
                    else:
                        X=np.vstack((key,X))
                    # fetch key presses for current frame from log.json
                    key_map_boolean = state.get('presses')
                    # Convert "true" entries to 1 and "false" entries to 0
                    key_map_numeric = {k: 1 if key_map_boolean[k] else 0 for k in key_map_boolean}
                    key_map_numeric = pd.Series(key_map_numeric)
                    print(key_map_numeric[4])
                    if count==1:
                        if key_map_numeric[4]==1:
                            y[0]=1
                        elif key_map_numeric[6]==1:
                            y[0]=2
                        else:
                            y[0]=0
                    else:
                        if key_map_numeric[4]==1:
                            y=np.concatenate([y,[1]])
                        elif key_map_numeric[6]==1:
                            y=np.concatenate([y,[2]])
                        else:
                            y=np.concatenate([y,[0]])

                re_sampler = RandomOverSampler(random_state=60)
                X, y = re_sampler.fit_sample(X, y)
                self.trained_model=nn_model.fit(X,y)

        except FileNotFoundError:
            logger.warning("Screenshot not found, skipping training data.")

    def pickle_model(self):
        """ Save model as a pickled file in models/ """
        model_output = os.path.join(self.models_dir, "{}.pickle".format(self.model_name))
        with open(model_output, 'wb') as m:
            pfile = {"model": self.trained_model}
            pickle.dump(pfile, m, pickle.HIGHEST_PROTOCOL)