import pickle
from src import screen, helper, downsampling
import os
import random
class NaiveAgent:
    def __init__(self, pickled_model_path):
        saved_file = open(pickled_model_path, 'rb')
        saved_obj = pickle.load(saved_file)
        self.model = saved_obj.get("model")
        self.defaults = saved_obj.get("defaults")
        saved_file.close()
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.game_name = 'NABE01'
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
