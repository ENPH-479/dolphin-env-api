""" This module provides tools for programmatic control of the Dolphin emulator's screenshot functionality. """

import pyautogui
import logging

logger = logging.getLogger(__name__)


def get_screen():
    """ Use Dolphin's screenshot ability to take a screenshot of the current Dolphin emulator state.

    Note:
        By default, the resulting screenshots are saved to:
        ~/.dolphin-emu/ScreenShots
    """
    try:
        pyautogui.keyDown('f9')
        pyautogui.keyUp('f9')
    except (AssertionError, TypeError):
        logger.exception("Error taking screenshot")
        # TODO track which screenshots failed
