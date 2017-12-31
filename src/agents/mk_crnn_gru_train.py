"""
This module implements a Convolutional Recurrent Neural Network (CRNN) Mario Kart AI Agent using
Gated Recurrent Units (GRUs) and PyTorch.
"""

import logging
import os
import matplotlib.pyplot as plt

from src import helper, keylog
from src.agents.train_valid_data import get_mario_train_valid_loader

import torch
import torch.nn as nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)

# Hyper Parameters
input_size = 15
hidden_size_1 = 64
hidden_size_2 = 64
hidden_size_3 = 64
output_vec = len(keylog.Keyboard)
history = 5
num_epochs = 50
batch_size = 50
l2_reg = 0.05
learning_rate = 1e-5


class MKCRNN_gru(nn.Module):
    def __init__(self):
        """ Convolutional Recurrent Neural Network (CRNN) architecture of Mario Kart AI agent with GRUs. """
        super(MKCRNN_gru, self).__init__()
        self.input_size = input_size
        self.conv1_in, self.conv1_out, self.conv1_kernel = history, 9, 3
        self.conv1_max_kernel = 3
        self.conv2_in, self.conv2_out, self.conv2_kernel = self.conv1_out, history, 3
        self.gru_in = self.conv2_out * 5 * 5
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_vec = output_vec

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.conv1_in, self.conv1_out, kernel_size=self.conv1_kernel,
                      stride=1, padding=1),  # output shape (self.out_channels, 15, 15)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=self.conv1_max_kernel),  # output shape (self.out_channels, 5, 5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv2_in, self.conv2_out, kernel_size=self.conv2_kernel, padding=1, stride=1),
            nn.LeakyReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(history * self.hidden_size_1, self.hidden_size_2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size_2, self.hidden_size_3),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size_3, self.output_vec)
        )
        self.gru = nn.GRU(5*5,
                            self.hidden_size_1,
                            num_layers=history, dropout=0.5)

        if torch.cuda.is_available():
            print("cuda is available.")
            self.cuda()

    def forward(self, x):
        """ Forward pass of the neural network. Accepts a tensor of size input_size*input_size. """
        x = x.view(-1, self.conv1_in, self.input_size, self.input_size)
        x = Variable(x).float()
        if torch.cuda.is_available():
            x = x.cuda()
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(-1,history,5*5)
        out, _ = self.gru(out)
        out = out.view(-1, history * self.hidden_size_1)
        out = self.encoder(out)
        return out


if __name__ == '__main__':
    """ Train neural network Mario Kart AI agent. """
    mkcrnn = MKCRNN_gru()

    # Define gradient descent optimizer and loss function
    optimizer = torch.optim.Adam(mkcrnn.parameters(), weight_decay=l2_reg, lr=learning_rate)
    loss_func = nn.MSELoss()

    # Load data
    train_loader, valid_loader = get_mario_train_valid_loader(batch_size, False, 123, history=history)

    # Store validation losses
    validation_losses = []

    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(train_loader):
            # Wrap label 'y' in variable
            y_label = y.view(-1, output_vec)
            if torch.cuda.is_available():
                y_label = y_label.cuda()
            nn_label = Variable(y_label)

            # Forward pass
            forward_pass = mkcrnn(x)
            loss = loss_func(forward_pass, nn_label)  # compute loss
            optimizer.zero_grad()  # zero gradients from previous step
            loss.backward()  # compute gradients
            optimizer.step()  # apply backpropagation

            # Log training data
            if step % 50 == 0:
                print('Epoch: ', epoch, 'Step: ', step, '| training loss: %.4f' % loss.data[0])
                valid_loss = 0
                for (valid_x, valid_y) in valid_loader:
                    valid_y_label = valid_y.view(-1, output_vec)
                    if torch.cuda.is_available():
                        valid_y_label = valid_y_label.cuda()
                    valid_nn_label = Variable(valid_y_label)
                    valid_forward_pass = mkcrnn(valid_x)
                    valid_loss_eval = loss_func(valid_forward_pass, valid_nn_label)  # compute validation loss
                    valid_loss += valid_loss_eval.data[0]
                print('Epoch: ', epoch, 'Step: ', step, '| validation loss: %.4f' % valid_loss)
                validation_losses.append(valid_loss)

    # Save model
    torch.save(mkcrnn, os.path.join(helper.get_models_folder(), "mkcrnn_{}_gru.pkl".format(history)))

    # Save validation curve data
    fig_data = [validation_losses, history, num_epochs, batch_size, learning_rate]
    helper.pickle_object(fig_data, "mkcrnn_training_data_{}_gru".format(history))

    # Plot validation curve
    f = plt.figure()
    plt.plot(validation_losses)
    plt.ylabel('Validation Error')
    plt.xlabel('Number of Iterations')
    plt.title('GRU CRNN Cross Validation Error, Learning Rate = %s, Batch Size = %i, Number of Epochs = %i' % (
        learning_rate, batch_size, num_epochs))
    plt.show(block=True)
