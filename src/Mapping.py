import enum
import logging
import json
import cv2
import os
from pynput import keyboard

from src import dolphinscreenshot, helper

logger = logging.getLogger(__name__)


class Mapper():
    def __init__(self):
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.file_name="log.json"
        self.output_dir = helper.get_output_folder()
        self.image_dir = os.path.join(helper.get_output_folder(), "images")

    def mapping(self):
        num_files = len(os.listdir(self.image_dir))
        count = 1
        state_map=dict()
        with open(os.path.join(self.output_dir, self.file_name), 'r') as f:
            log=json.load(f)

            while count <= num_files:
                file_name = "{}.png".format(count)
                ima = cv2.imread(os.path.join(self.image_dir, file_name))
                ima_tuple = tuple(ima)


                key_map_boolean = log['data'][count]['presses']
                key_map_numeric=dict()

                # Convert "true" to 1 and "false" to 0
                for k in key_map_boolean:
                    if key_map_boolean['k'] == "true":
                        key_map_numeric['k'] = 1

                    elif key_map_boolean == "false":
                        key_map_numeric['k'] = 0

                # Check if the state already exists in the state map
                if ima_tuple in state_map:

                    for k in state_map['ima_tuple']:
                        if key_map_numeric['k'] == 1:
                            state_map['ima_tuple']['k'] +=1

                else:
                    state_map[ima_tuple] = key_map_numeric

            count += 1



test=Mapper
test.mapping()

'''
with open(os.path.join(helper.get_output_folder(), "log.json"), 'r') as f:
    log=json.load_dolphin_state(f)

    print(iter(log['data'][1]['presses']))
'''