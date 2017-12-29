import logging
import os
import matplotlib.pyplot as plt

from src import helper, keylog
from src.agents.train_valid_data import get_mario_train_valid_loader

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters
input_size = 15
output_vec = len(keylog.Keyboard)

num_epochs = 100
batch_size = 50
l2_reg = 0.05
learning_rate = 1e-5


class MKCNN(nn.Module):
    def __init__(self):
        """ Convolutional neural network architecture of Mario Kart AI agent. """
        super(MKCNN, self).__init__()
        self.input_size = input_size
        self.conv1_in, self.conv1_out, self.conv1_kernel = 2, 9, 3
        self.conv1_max_kernel = 3
        self.conv2_in, self.conv2_out, self.conv2_kernel = self.conv1_out, 6, 3
        self.fc_hidden1 = self.conv2_out * 5 * 5
        self.fc_hidden2 = 64
        self.fc_hidden3 = 64

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
        self.fc = nn.Sequential(
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.fc_hidden2, self.fc_hidden3),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.fc_hidden3, output_vec),
        )

        if torch.cuda.is_available():
            print("cuda is available.")
            self.cuda()

    def forward(self, x):
        """ Forward pass of the neural network. Accepts a tensor of size input_size*input_size. """
        # print(x.shape)
        x = x.view(-1, self.conv1_in, self.input_size, self.input_size)
        # print(x.shape)
        x = Variable(x).float()
        if torch.cuda.is_available():
            x = x.cuda()
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)  # flatten the output of conv2 to (batch_size, 6 * 5 *5)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        return out


if __name__ == '__main__':
    """ Train neural network Mario Kart AI agent. """
    mkcnn = MKCNN()
    # define gradient descent optimizer and loss function
    optimizer = torch.optim.Adam(mkcnn.parameters(), weight_decay=l2_reg, lr=learning_rate)
    loss_func = nn.MSELoss()

    # load data
    train_loader, valid_loader = get_mario_train_valid_loader(batch_size, False, 123, history=2)
    # store validation losses
    validation_losses = []

    for epoch in range(num_epochs):

        for step, (x, y) in enumerate(train_loader):
            # Wrap label 'y' in variable
            y_label = y.view(-1, output_vec)
            if torch.cuda.is_available():
                y_label = y_label.cuda()
            nn_label = Variable(y_label)
            # forward pass
            forward_pass = mkcnn(x)
            loss = loss_func(forward_pass, nn_label)  # compute loss
            optimizer.zero_grad()  # zero gradients from previous step
            loss.backward()  # compute gradients
            optimizer.step()  # apply backpropagation

            # log training
            if step % 50 == 0:
                print('Epoch: ', epoch, 'Step: ', step, '| train loss: %.4f' % loss.data[0])
                valid_loss = 0
                for (valid_x, valid_y) in valid_loader:
                    valid_y_label = valid_y.view(-1, output_vec)
                    if torch.cuda.is_available():
                        valid_y_label = valid_y_label.cuda()
                    valid_nn_label = Variable(valid_y_label)

                    valid_forward_pass = mkcnn(valid_x)
                    valid_loss_eval = loss_func(valid_forward_pass, valid_nn_label)  # compute validation loss
                    valid_loss += valid_loss_eval.data[0]
                print('Epoch: ', epoch, 'Step: ', step, '| validation loss: %.4f' % valid_loss)
                validation_losses.append(valid_loss)

    # save model
    torch.save(mkcnn, os.path.join(helper.get_models_folder(), "mkcnn.pkl"))

    # show validation curve
    f = plt.figure()
    plt.plot(validation_losses)
    plt.ylabel('Validation error')
    plt.xlabel('Number of iterations')
    plt.title('CNN Cross Validation Error, learning rate = %s, batch size = %i, number of Epochs= %i' % (
        learning_rate, batch_size, num_epochs))
    plt.show()
