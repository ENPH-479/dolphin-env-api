""" Module containing miscellaneous helper functions. """

import re
import os

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


def get_home_folder():
    return os.path.expanduser('~')


def generate_img_key(image):
    try:
        image_single_channel = image[:, :, 1]
    except IndexError:
        image_single_channel = image
    return tuple(image_single_channel.flatten())
