import logging

from src import pad

# configure logger
logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# temp testing function
def main():
    with pad.Pad("~/.dolphin-emu/Pipes/pipe") as p:
        p.press_button(pad.Button.A)
        p.press_button(pad.Button.START)
        p.reset()


if __name__ == '__main__':
    main()
