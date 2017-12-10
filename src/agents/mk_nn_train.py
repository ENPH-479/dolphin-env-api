""" This module contains code for maintaining the basic Mario Kart AI agent's state-decision map. """
import json
import logging
import os
import cv2
from torch.utils.data import Dataset, DataLoader

from src import helper, keylog

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters
input_size = 15
hidden_size_1 = 256
hidden_size_2 = 64
hidden_size_3 = 24
output_vec = len(keylog.Keyboard)

num_epochs = 5
batch_size = 3


# learning_rate = 0.01


class MKRNN(nn.Module):
    def __init__(self):
        """ Neural network architecture of Mario Kart AI agent. """
        super(MKRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_vec = output_vec

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, self.hidden_size_1),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size_2, self.hidden_size_3),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size_3, self.output_vec)
        )

    def forward(self, x):
        """ Forward pass of the neural network. Accepts a tensor of size input_size*input_size. """
        x = Variable(x.view(-1, self.input_size * self.input_size)).float()
        encoded = self.encoder(x)
        return encoded


class MarioKartDataset(Dataset):
    """Mario Kart dataset."""

    def __init__(self, log_file=os.path.join(helper.get_dataset_folder(), 'mario_kart', "mario_kart.json")):
        """
        Args:
            log_file: file path location to the log file containing key press states.
        """
        self.images = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")
        with open(log_file, 'r') as f:
            self.data = json.load(f).get('data')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key_state = self.data[idx]
        img_name = os.path.join(self.images, "{}.png".format(key_state.get('count')))
        image = cv2.imread(img_name)
        tensor = helper.get_tensor(image)
        return tensor, helper.get_output_vector(key_state['presses'])


if __name__ == '__main__':
    """ Train neural network Mario Kart AI agent. """
    mkrnn = MKRNN()
    # define gradient descent optimizer and loss function
    optimizer = torch.optim.Adam(mkrnn.parameters())
    loss_func = nn.MSELoss()

    # data_path = os.path.join(helper.get_dataset_folder(), 'mario_kart', "mario_kart.json")
    # image_dir = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")
    # with open(data_path, 'r') as keylog_file:
    #     key_log_data = json.load(keylog_file).get('data')

    # load data
    dataset = MarioKartDataset()
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(loader):
            # Wrap label 'y' in variable
            nn_label = Variable(y.view(-1, output_vec))

            # forward pass
            forward_pass = mkrnn(x)

            loss = loss_func(forward_pass, nn_label)  # compute loss
            optimizer.zero_grad()  # zero gradients from previous step
            loss.backward()  # compute gradients
            optimizer.step()  # apply backpropagation

            # log training
            if step % 100 == 0:
                print('Epoch: ', epoch, 'Step: ', step, '| train loss: %.4f' % loss.data[0])

    # save model
    torch.save(mkrnn, os.path.join(helper.get_models_folder(), "mkrnn.pkl"))
