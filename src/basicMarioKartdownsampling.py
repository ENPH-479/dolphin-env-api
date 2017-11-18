""" This module implements the image downsampling component of the basic Mario Kart AI agent. """

import logging
import traceback
import cv2
import numpy as np
import os
from skimage.measure import block_reduce
from src import helper

logger = logging.getLogger(__name__)


class Downsampler:
    def __init__(self, game_name, blur_size=25, intermediate_dim=300, final_dim=10):
        self.game_name = game_name
        self.blur_size = blur_size
        if intermediate_dim % final_dim != 0:
            raise ValueError("final dimensions must divide into intermediate dimensions completely.")
        self.pooling_dim = int(intermediate_dim / final_dim)
        self.inter_dim = (intermediate_dim, intermediate_dim)
        self.screenshot_dir = os.path.join(helper.get_home_folder(), '.dolphin-emu', 'ScreenShots')
        self.output_dir = os.path.join(helper.get_output_folder(), "images")

    def downsample_dir(self, save_imgs=False, clean_data=False):
        try:
            screen_dir = os.path.join(self.screenshot_dir, self.game_name)
            num_files = len(os.listdir(screen_dir))
            count = 1
            while count <= num_files:
                file_name = "{}-{}.png".format(self.game_name, count)
                f = os.path.join(screen_dir, file_name)
                self.downsample(f, output_name=count, save_img=save_imgs)
                if clean_data: os.unlink(f)
                count += 1
        except:
            logger.error(traceback.format_exc())
            logger.error("failed to downsample entire screenshot directory.")

    def downsample(self, file_path, output_name=None, save_img=False):
        try:
            logger.info("downsampling {}".format(file_path))
            img = cv2.imread(file_path)
            # Filtering
            im_blurred = cv2.medianBlur(img, self.blur_size)
            # Resizing
            im_rs = cv2.resize(im_blurred, self.inter_dim)
            # Grayscaling
            im_gray = cv2.cvtColor(im_rs, cv2.COLOR_BGR2GRAY)
            # Maxpooling
            data = np.asarray(im_gray, dtype='int32')
            img_pooled = block_reduce(data, block_size=(self.pooling_dim, self.pooling_dim), func=np.max)

            # Quantization
            quantized = img_pooled
            quantized[(quantized > 0) & (quantized < 51)] = 21
            quantized[(quantized > 50) & (quantized < 103)] = 78
            quantized[(quantized > 102) & (quantized < 155)] = 130
            quantized[(quantized > 154) & (quantized < 207)] = 182
            quantized[(quantized > 206) & (quantized < 256)] = 232

            if save_img: self.save_image(quantized, output_name)
            return quantized
        except:
            logger.error(traceback.format_exc())
            logger.error("failed to downsample image.")

    def save_image(self, img, output_name):
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.output_dir, '{}.png'.format(output_name)), img)
