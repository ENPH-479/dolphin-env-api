import logging

import pyautogui

from src import helper

logger = logging.getLogger(__name__)


def save(slot_key):
    try:
        helper.validate_function_key(slot_key)
        pyautogui.keyDown('shift')
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
        pyautogui.keyUp('shift')
    except (AssertionError, TypeError):
        logger.exception("Error saving state {}:".format(slot_key))


def load(slot_key):
    try:
        helper.validate_function_key(slot_key)
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error loading state {}:".format(slot_key))

load('f3')
