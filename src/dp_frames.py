""" This module provides tools for programmatic control of the Dolphin emulator's frame control functionality. """

import logging
import pyautogui

logger = logging.getLogger(__name__)


def advance(slot_key):
    """ Advance the state of the running Dolphin emulator by a single frame.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin emulator's
            frame advance hot-key. P is often the default key.
    """
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error advancing frame {}:".format(slot_key))


def inc_speed(slot_key):
    """ Increase the frame advance speed of the running Dolphin emulator.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin emulator's increase
            frame speed hot-key. M is often the default key.
    """
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error increasing speed {}:".format(slot_key))


def dec_speed(slot_key):
    """ Decrease the frame advance speed of the running Dolphin emulator.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin emulator's decrease
            frame speed hot-key. N is often the default key.
    """
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error decreasing speed {}:".format(slot_key))


def res_speed(slot_key):
    """ Reset the frame advance speed of the running Dolphin emulator to the default speed.

    Args:
        slot_key: Keyboard key corresponding to the Dolphin emulator's reset
            frame speed hot-key. R is often the default key.
    """
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error resetting speed {}:".format(slot_key))
