import logging

import pyautogui

from src import helper

logger = logging.getLogger(__name__)

def advance(slot_key): #P
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error advancing frame {}:".format(slot_key))

def inc_speed(slot_key): #M
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error increasing speed {}:".format(slot_key))

def dec_speed(slot_key): #N
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error decreasing speed {}:".format(slot_key))

def res_speed(slot_key): #R
    try:
        pyautogui.keyDown(slot_key)
        pyautogui.keyUp(slot_key)
    except (AssertionError, TypeError):
        logger.exception("Error reseting speed {}:".format(slot_key))