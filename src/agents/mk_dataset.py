import json
import logging
import os
import cv2
import torch
from torch.utils.data import Dataset

from src import helper

logger = logging.getLogger(__name__)


class MarioKartDataset(Dataset):
    """Mario Kart dataset."""

    def __init__(self, log_file=os.path.join(helper.get_dataset_folder(), 'mario_kart', "mario_kart.json"),
                 transform=None, history=1):
        """
        Args:
            log_file: file path location to the log file containing key press states.
        """
        self.images = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")
        with open(log_file, 'r') as f:
            self.data = json.load(f).get('data')
        self.transform = transform
        self.history = history

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key_state = self.data[idx]
        img_count = key_state.get('count')
        tensors = []
        for i in range(self.history):
            img_name = os.path.join(self.images, "{}.png".format(img_count - i))
            image = cv2.imread(img_name)
            img_tensor = helper.get_tensor(image)
            if img_tensor is not None:
                tensors.append(img_tensor)
            else:
                logger.warning("Image not found, using last tensor found")
                print(img_count)
                tensors.append(tensors[-1])
        tensor = torch.stack(tensors)
        return tensor, helper.get_output_vector(key_state['presses'])
