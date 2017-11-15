import enum
import logging
import json
import threading

import os
from pynput import keyboard

from src import screen

logger = logging.getLogger(__name__)


@enum.unique
class Keyboard(enum.Enum):
    none = 0  # Neutral controller state
    x = 1  # PRESS A
    z = 2  # PRESS B
    c = 3  # PRESS X
    s = 4  # PRESS Y
    d = 5  # PRESS Z
    left = 6  # PRESS LEFT
    right = 7  # PRESS RIGHT
    up = 8  # PRESS UP
    down = 9  # PRESS DOWN
    enter = 10  # PRESS ENTER


class KeyLog():
    def __init__(self, logging_freq=0.3):
        self.state = dict((el.name, False) for el in Keyboard)
        self.count = 1
        self.log = {"data": []}
        self.finish = False
        self.logging_freq = logging_freq

    # Collect events until released
    def start(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            self.record()
            listener.join()
            logger.info("Key logging session finished.")

    def on_press(self, key):
        try:
            value = self._get_key_value(key)
            if not self.state[Keyboard[value].name]:
                self.state[Keyboard[value].name] = True
                logger.debug('key {} pressed {}'.format(value, self.state[Keyboard[value].name]))
        except KeyError:
            logger.warning("Pressed key not defined within allowable controller inputs.")

    def on_release(self, key):
        try:
            value = self._get_key_value(key)
            if self.state[Keyboard[value].name]:
                self.state[Keyboard[value].name] = False
                logger.debug('{} released {}'.format(value, self.state[Keyboard[value].name]))
        except KeyError:
            if key == keyboard.Key.esc:
                self.finish = True
                # save key press log
                self.save_to_file()
                # Stop listener
                return False

    def record(self):
        if self.finish: return
        threading.Timer(self.logging_freq, self.record).start()
        screen.get_screen()
        self.log['data'].append({
            "count": self.count,
            "presses": dict(self.state)
        })
        self.count += 1

    def save_to_file(self):
        curr_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(curr_path, "../output")
        os.makedirs(file_path, exist_ok=True)
        with open(os.path.join(file_path, 'log.json'), 'w') as f:
            json.dump(self.log, f)

    @staticmethod
    def _get_key_value(key):
        try:
            # regular key
            return key.char
        except AttributeError:
            # special key
            return key.name
