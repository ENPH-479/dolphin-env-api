import logging

import time

from src import dolphincontroller, keylog, basicMarioKartdownsampling

#configure logger
#logging.basicConfig(format='%(name)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
#logger = logging.getLogger(__name__)


# temp testing function
def main():
    # with pad.DolphinController("~/.dolphin-emu/Pipes/pipe") as p:
    #     p.press_release_button(pad.Button.START)
    #     time.sleep(0.1)

    downsampling.Downsampler('NABE01', final_dim=15).downsample_dir(save_imgs=True)

    #k = keylog.KeyLog()
    #k.start()


if __name__ == '__main__':
    main()
