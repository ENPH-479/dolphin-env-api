""" This module provides tools for programmatic control of the Dolphin emulator's save state functionality. """

import logging
import pyautogui
from src import helper

logger = logging.getLogger(__name__)


def save_dolphin_state(slot_key):
    """ Save the current state of a running Dolphin emulator.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin save state slot
            to save the current state of the emulator to.
    """
    try:
        helper.validate_function_key(slot_key)
        pyautogui.keyDown('shift')
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
        pyautogui.keyUp('shift')
    except (AssertionError, TypeError):
        logger.exception("Error saving state {}:".format(slot_key))


def load_dolphin_state(slot_key):
    """ Load a saved save state to a running Dolphin emulator.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin save state slot
            to load from in the running Dolphin emulator.
    """
    try:
        helper.validate_function_key(slot_key)
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error loading state {}:".format(slot_key))
