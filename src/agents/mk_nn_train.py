""" This module contains code for maintaining the basic Mario Kart AI agent's state-decision map. """
import json
import logging
import os
import cv2

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
        x = Variable(x.view(self.input_size * self.input_size)).float()
        encoded = self.encoder(x)
        return encoded


if __name__ == '__main__':
    """ Train neural network Mario Kart AI agent. """
    mkrnn = MKRNN()
    # define gradient descent optimizer and loss function
    optimizer = torch.optim.Adam(mkrnn.parameters())
    loss_func = nn.MSELoss()

    data_path = os.path.join(helper.get_output_folder(), "data.json")
    image_dir = os.path.join(helper.get_output_folder(), "images")
    with open(data_path, 'r') as keylog_file:
        key_log_data = json.load(keylog_file).get('data')

    for epoch in range(num_epochs):
        for step, state in enumerate(key_log_data):
            count = state.get('count')
            # Read the image data
            image_file_name = "{}.png".format(count)
            img_path = os.path.join(image_dir, image_file_name)
            image_data = cv2.imread(img_path)

            # Generate input tensor from image.
            x = helper.get_tensor(image_data)
            # Get output tensor of key presses
            y = helper.get_output_vector(state.get('presses'))
            # Wrap in variable
            nn_label = Variable(y.view(output_vec))

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
