#!/usr/bin/env python

""" This module contains some tests of code functionality. """

import logging
import os
import time

import cv2
import torch

from src import dp_controller, keylog, mk_downsampler, key2pad, helper, dataset_merger
from src.agents import state_model, mk_naive_agent, mk_nn
from src.agents.mk_nn_train import MKRNN
from src.agents.mk_cnn_train import MKCNN

# Configure logger
logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dolphin_controller():
    """ Check that the Dolphin controller can communicate with Dolphin using a fifo pipe. """
    with dp_controller.DolphinController("~/.dolphin-emu/Pipes/pipe") as p:
        p.press_release_button(dp_controller.Button.START, delay=0.1)
        time.sleep(0.1)


def test_key_logging():
    """ Check that keyboard input can be successfully logged to a .json file. """
    k = keylog.KeyLog()
    k.start()


def test_mario_kart_downsampler():
    """ Check that the basic Mario Kart image downsampler can process images saved by Dolphin. """
    mk_downsampler.Downsampler('NABE01', final_dim=15).downsample_dir(save_imgs=True)


def test_state_map_population():
    """ Check that a state decision map can be properly populated from images and key logs. """
    model = state_model.StateModel(clean_imgs=True)
    model.train()
    print(model.defaults)
    print(len(model.state_decision_map))


def test_key2pad():
    temp = {'left': True, 'right': False, 'up': False, 's': False, 'none': False, 'x': False, 'c': False,
            'enter': False, 'down': True, 'd': False, 'z': False}
    test = key2pad.KeyPadMap()
    test.update(temp)
    print(test.previous_keys)
    temp2 = {'left': False, 'right': True, 'up': True, 's': True, 'none': True, 'x': True, 'c': True, 'enter': False,
             'down': True, 'd': False, 'z': False}
    test.update(temp2)
    print(test.previous_keys)


def test_process_frame():
    """ Check that the basic Mario Kart AI can process a Dolphin screenshot and choose an action. """
    agent = mk_naive_agent.MarioKartAgent(os.path.join(helper.get_models_folder(), "naive_model.pickle"))
    while True:
        agent.process_frame()


def test_nn_single_imge():
    model = torch.load(os.path.join(helper.get_models_folder(), "mkrnn.pkl"))

    image_dir = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")

    image_file_name = "140.png"
    img_path = os.path.join(image_dir, image_file_name)
    image_data = cv2.imread(img_path)

    # Generate tensor from image
    x = helper.get_tensor(image_data)

    pred = model(x)
    key_state = helper.get_key_state_from_vector(pred)
    print(pred, key_state)


def test_nn(nn_name):
    """ Check that neural network Mario Kart AI can process a Dolphin screenshot and choose an action. """
    agent = mk_nn.MarioKartNN(os.path.join(helper.get_models_folder(), nn_name))
    while True:
        agent.process_frame()


def log_downsample_merge(logging_delay=0.3):
    """ Log key inputs, downsample images, merge to main dataset. """
    k = keylog.KeyLog(logging_delay)
    k.start()
    mk_downsampler.Downsampler('NABE01', final_dim=15).downsample_dir(save_imgs=True)
    dataset_merger.merge('mario_kart')


# Main function for entering tests
def main():
    # test_dolphin_controller
    # test_key_logging()
    # test_mario_kart_downsampler()
    # test_state_map_population()
    # test_key2pad()
    # test_process_frame()
    # test_nn_single_imge()
    # log_downsample_merge(logging_delay=0.2)
    test_nn("mkcnn.pkl")


if __name__ == '__main__':
    main()
