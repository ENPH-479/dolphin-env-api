#!/usr/bin/env python

""" This module contains some tests of code functionality. """

import logging
import time
import os
from src import dp_controller, keylog, mk_downsampler, state_model, key2pad, mk_naive_agent, helper

# Configure logger
logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dolphin_controller():
    """ Check that the Dolphin controller can communicate with Dolphin using a fifo pipe. """
    with dp_controller.DolphinController("~/.dolphin-emu/Pipes/pipe") as p:
        p.press_release_button(dp_controller.Button.START, delay=0.1)
        time.sleep(0.1)


def test_mario_kart_downsampler():
    """ Check that the basic Mario Kart image downsampler can process images saved by Dolphin. """
    mk_downsampler.Downsampler('NABE01', final_dim=15).downsample_dir(save_imgs=True)


def test_key_logging():
    """ Check that keyboard input can be successfully logged to a .json file. """
    k = keylog.KeyLog()
    k.start()


def test_state_map_population():
    """ Check that a state decision map can be properly populated from images and key logs. """
    model = state_model.StateModel()
    model.train()
    print(model.state_decision_map)
    print(model.defaults)
    print(model.state_counts)
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
    agent = mk_naive_agent.MarioKartAgent(os.path.join(helper.get_models_folder(), "naive_model.pickle" ), 'NABE01')
    while True:
        agent.process_frame()


# Main function for entering tests
def main():
    # test_dolphin_controller
    # test_mario_kart_downsampler()
    # test_key_logging()
    # test_state_map_population()
    test_key2pad()

if __name__ == '__main__':
    main()
