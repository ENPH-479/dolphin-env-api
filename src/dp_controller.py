""" This module provides tools for programmatic control of the Dolphin emulator via fifo pipes.

Note: More information about the Dolphin emulator can be found at:
    https://dolphin-emu.org/
    https://github.com/dolphin-emu/dolphin
"""

import enum
import logging
import os
import time

logger = logging.getLogger(__name__)


@enum.unique
class Button(enum.Enum):
    """ Supported Dolphin buttons """
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11


@enum.unique
class Trigger(enum.Enum):
    """ Supported Dolphin triggers """
    L = 0
    R = 1


@enum.unique
class Stick(enum.Enum):
    """ Supported Dolphin  sticks """
    MAIN = 0
    C = 1


class DolphinController:
    """ Class allowing user to send controller inputs to Dolphin programmatically using fifo pipes.

    Note:
        Currently only functional on Linux operating systems. See below for more info:
        https://wiki.dolphin-emu.org/index.php?title=Pipe_Input
    """

    def __init__(self, path):
        """ Create DolphinController instance, but do not open the fifo pipe.

        Args:
            path: Location of the Dolphin fifo pipe configuration file

        Note:
            The Dolphin fifo pipe configuration file is often located at:
            ~/.dolphin-emu/Pipes/pipe
        """
        self.pipe = None
        self.path = os.path.expanduser(path)
        logger.info("Controller pad initialized.")

    def __enter__(self):
        """ Open the fifo pipe. Blocks until the other side is listening. """
        self.pipe = open(self.path, 'w', buffering=1)
        logger.info("Pipe opened.")
        return self

    def __exit__(self, *args):
        """ Close the fifo pipe. """
        if self.pipe:
            self.pipe.close()
            logger.info("Pipe closed.")

    def press_button(self, button):
        """ Press a Dolphin controller button.

        Args:
            button: The Dolphin controller button to press. Must be a supported Dolphin button.
        """
        assert button in Button
        self.pipe.write('PRESS {}\n'.format(button.name))
        logger.info("\n {} pressed".format(button.name))

    def release_button(self, button):
        """ Release a Dolphin controller button.

        Args:
            button: The Dolphin controller button to release. Must be a supported Dolphin button.
        """
        assert button in Button
        self.pipe.write('RELEASE {}\n'.format(button.name))
        logger.info("\n {} released".format(button.name))

    def press_release_button(self, button, delay):
        """ Press and release a Dolphin controller button.

        Args:
            button: The Dolphin controller button to release. Must be a supported Dolphin button.
            delay: The number of seconds to delay between pressing and releasing the button.
        """
        self.press_button(button)
        time.sleep(delay)
        self.release_button(button)

    def set_trigger(self, trigger, amount):
        """ Set how far down a Dolphin controller trigger is pressed.

        Args:
            trigger: The Dolphin controller trigger to set. Must be a supported Dolphin trigger.
            amount: How far to press the Dolphin controller trigger. Must be a value between 0 and 1.
                0 indicates the trigger is released, and 1 indicates the controller is fully pressed
                down.
        """
        assert trigger in Trigger
        assert 0 <= amount <= 1
        self.pipe.write('SET {} {:.2f}\n'.format(trigger.name, amount))

    def set_stick(self, stick, x, y):
        """ Set the location of a Dolphin controller stick/joystick.

        Args:
            stick: The Dolphin controller stick to set. Must be a supported Dolphin stick.
            x: The x position of the stick. Must be a value between 0 and 1. 0.5 indicates the trigger
                is neutral, 1 is full up, and -1 is full down.
            y: The y position of the stick. Must be a value between 0 and 1. 0.5 indicates the trigger
                is neutral, 1 is full up, and -1 is full down.
        """
        assert stick in Stick
        assert 0 <= x <= 1 and 0 <= y <= 1
        self.pipe.write('SET {} {:.2f} {:.2f}\n'.format(stick.name, x, y))

    def reset(self):
        """ Reset all Dolphin controller elements to released or neutral position. """
        for button in Button:
            self.release_button(button)
        for trigger in Trigger:
            self.set_trigger(trigger, 0)
        for stick in Stick:
            self.set_stick(stick, 0.5, 0.5)
