import json
import logging
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src import helper, keylog
from src.agents.train_valid_data import get_mario_train_valid_loader
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
# Hyper Parameters
input_size = 15
hidden_size_1 = 256
hidden_size_2 = 64
hidden_size_3 = 24
output_vec = len(keylog.Keyboard)

num_epochs = 3
batch_size = 5


class MKCNN(nn.Module):
    def __init__(self):
        """ Neural network architecture of Mario Kart AI agent. """
        super(MKCNN, self).__init__()
        self.input_size = input_size
        self.in_channels = 5
        self.out_channels = 6
        self.kernel_size = 5
        self.stride = 2
        self.padding = 1
        self.num_filters = 2
        self.output_vec = output_vec
        self.count=1
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,(self.kernel_size,self.kernel_size),stride=self.stride,padding=self.padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=(1,1), padding=1,stride=2),
            nn.LeakyReLU(),
            nn.Linear(5,10),
            nn.Softmax(),

        )

    def forward(self, x):
        """ Forward pass of the neural network. Accepts a tensor of size input_size*input_size. """
        #x = Variable(x.view(-1, self.input_size * self.input_size)).float()
        #self.count+=1
        #print(self.count)
        #print(x)
        encoded = self.encoder(Variable(x))
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
    mkcnn = MKCNN()
    # define gradient descent optimizer and loss function
    optimizer = torch.optim.Adam(mkcnn.parameters(), weight_decay=0.05, lr=1e-4)
    loss_func = nn.MSELoss()

    # data_path = os.path.join(helper.get_dataset_folder(), 'mario_kart', "mario_kart.json")
    # image_dir = os.path.join(helper.get_dataset_folder(), 'mario_kart', "images")
    # with open(data_path, 'r') as keylog_file:
    #     key_log_data = json.load(keylog_file).get('data')

    # load data
    train_loader, valid_loader = get_mario_train_valid_loader(batch_size, False, 123)
    # store validation losses
    validation_losses = []

    for epoch in range(num_epochs):

        for step, (x, y) in enumerate(train_loader):
            # Wrap label 'y' in variable, y.shape=3,10
            nn_label = Variable(y.view(-1, output_vec))
            # forward pass
            x=x[None,:,:,]
            x=x.float()
            forward_pass = mkcnn(x)
            forward_pass=forward_pass[0,0,:,:]
            loss = loss_func(forward_pass, nn_label)  # compute loss
            optimizer.zero_grad()  # zero gradients from previous step
            loss.backward()  # compute gradients
            optimizer.step()  # apply backpropagation

            # log training
            if step % 50 == 0:
                print('Epoch: ', epoch, 'Step: ', step, '| train loss: %.4f' % loss.data[0])
                valid_loss = 0
                for (valid_x, valid_y) in valid_loader:
                    valid_nn_label = Variable(valid_y.view(-1, output_vec))
                    valid_x = valid_x.unsqueeze(0)
                    valid_x = valid_x.float()
                    #print(valid_x.shape)
                    valid_forward_pass = mkcnn(valid_x)
                    valid_forward_pass = valid_forward_pass[0,0,:,:]
                    valid_loss_eval = loss_func(valid_forward_pass, valid_nn_label)  # compute validation loss
                    valid_loss += valid_loss_eval.data[0]
                print('Epoch: ', epoch, 'Step: ', step, '| validation loss: %.4f' % valid_loss)
                validation_losses.append(valid_loss)

    # save model
    torch.save(mkcnn, os.path.join(helper.get_models_folder(), "mkcnn.pkl"))

    # show validation curve
    f = plt.figure()
    plt.plot(validation_losses)
    plt.show()
