import logging

import time

from src import pad
from src import state
from src import frame

# configure logger
logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# temp testing function
def main():
    file = open("input.txt", "w")
    file.close()
    with pad.Pad("~/.dolphin-emu/Pipes/pipe") as p:

        p.prre_button(pad.Button.A)
        time.sleep(0.1)
        p.prre_button(pad.Button.START)
        time.sleep(0.1)


    state.load('f3')
    time.sleep(1)
    frame.advance('p')
    time.sleep(1)
    frame.advance('P')
    time.sleep(1)
    frame.advance('P')
    time.sleep(1)

if __name__ == '__main__':
    main()
