""" Module containing miscellaneous helper functions. """
import logging
import re
import os
import random

import torch

from src import keylog

logger = logging.getLogger(__name__)
fkey_pattern = re.compile(r'^F(\d{1,2})$', re.I)


def validate_function_key(slot_key):
    # check keyboard string
    m = fkey_pattern.match(slot_key)
    assert m
    # check function key is in 1-12 range
    assert 0 < int(m.group(1)) < 13


def get_output_folder():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_path, "../output")
    os.makedirs(file_path, exist_ok=True)
    return file_path


def get_models_folder():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_path, "../models")
    os.makedirs(file_path, exist_ok=True)
    return file_path


def get_dataset_folder():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_path, "../datasets")
    os.makedirs(file_path, exist_ok=True)
    return file_path


def get_home_folder():
    return os.path.expanduser('~')


def generate_img_key(image):
    try:
        image_single_channel = image[:, :, 1]
    except IndexError:
        image_single_channel = image
    except TypeError:
        logger.warning("Image key not generated.")
        return None
    return tuple(image_single_channel.flatten())


def get_tensor(image):
    try:
        image_single_channel = image[:, :, 1]
    except IndexError:
        image_single_channel = image
    except TypeError:
        logger.warning("Image tensor not generated.")
        return None
    return torch.from_numpy(image_single_channel).contiguous()


def get_output_vector(presses):
    out = []
    for key in keylog.Keyboard:
        out.append(presses.get(key.name, 0))
    return torch.FloatTensor(out)


def get_key_state_from_vector(vector):
    key_state = dict()
    for i, key in enumerate(keylog.Keyboard):
        rand_num = random.uniform(0, 1)
        key_state[key.name] = vector[i].data[0] > rand_num
    return key_state
