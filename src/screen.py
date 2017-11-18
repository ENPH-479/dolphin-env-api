import pyautogui
import logging

logger = logging.getLogger(__name__)

"""
Grab screenshot of current game state and save to ~/.dolphin-emu/ScreenShots
"""
def get_screen():
    try:
        pyautogui.keyDown('f9')
        pyautogui.keyUp('f9')
    except (AssertionError, TypeError):
        logger.exception("Error taking screenshot")
        # TODO track which screenshot failed
