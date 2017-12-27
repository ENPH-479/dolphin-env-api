import json
import os
import cv2
from torch.utils.data import Dataset

from src import helper


class MarioKartDataset(Dataset):
    """Mario Kart dataset."""

    def __init__(self, log_file=os.path.join(helper.get_dataset_folder(), 'mario_kart', "mario_kart.json"),
                 transform=None):
        """
        Args:
            log_file: file path location to the log file containing key press states.
        """
        self.images = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")
        with open(log_file, 'r') as f:
            self.data = json.load(f).get('data')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key_state = self.data[idx]
        img_name = os.path.join(self.images, "{}.png".format(key_state.get('count')))
        image = cv2.imread(img_name)
        # if self.transform is not None:
        #     image = self.transform(image)
        tensor = helper.get_tensor(image)
        return tensor, helper.get_output_vector(key_state['presses'])
