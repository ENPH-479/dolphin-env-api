import sys,os
import time
import pyautogui
import logging
logger = logging.getLogger(__name__)

script_dir=sys.path[0];
#file=open(os.path.join(script_dir,'~/.dolphin-emu/ScreenShots/NATE01/NATE01-1.png'),'r')
#file=open(os.path.join(os.path.expanduser('~'), '/home/lrs/.dolphin-emu/ScreenShots/NATE01/NATE01-1.png'))
#file.close()

#def screenshot():

'''
try:
    pyautogui.keyDown('f9')
    n = time.clock()
    pyautogui.keyUp('f9')
except (AssertionError, TypeError):
    logger.exception("Error reseting speed {}:".format(f9))

while os.path.isfile('/home/lrs/.dolphin-emu/ScreenShots/NATE01/NATE01-1.png')!=True:
    pass;

c=time.clock()
print(c-n)
'''
c=time.clock()
while time.clock()<c+60:
    try:
        pyautogui.keyDown('f9')
        pyautogui.keyUp('f9')
    except (AssertionError, TypeError):
        logger.exception("Error reseting speed {}:".format(f9))
