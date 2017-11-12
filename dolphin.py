import logging

import time

from src import pad, keylog

# configure logger
logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# temp testing function
def main():
    # with pad.Pad("~/.dolphin-emu/Pipes/pipe") as p:
    #     p.prre_button(pad.Button.START)
    #     time.sleep(0.1)
    k = keylog.KeyLog()
    k.start()


if __name__ == '__main__':
    main()
