#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib



class VAE(torch.nn.Module):
    def __init__(self, size):
        super().__init__()

        self.encoder = Encoder(size)


    def forward(self, data):
        means = self.encoder(data)

        # Get the height and width dimensions of the input data
        batch_size, _, height, width = data.size()

        means = means.view(1, 3, 1, 1)

        # Expand means values ​​to match the shape of the images
        constant_value = means.expand(batch_size, 3, height, width)
        
        return constant_value

    def getLoss(self):
        #        lossX = torch.nn.functional.mse_loss(res, inputs, reduction='sum')


        lossKL = torch.tensor(0.0)
        #        print(lossX, lossKL)
        #        loss = lossX + lossKL
        loss = lossKL
        return loss


class Encoder(torch.nn.Module):
    def __init__(self, size):
        super().__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(int(128 * (size[1] // 16) * (size[2] // 16)), 3)


    def forward(self, data):

        data = self.conv1(data)

        data = self.conv2(data)

        data = self.conv3(data)

        data = self.conv4(data)

        data = data.view(data.size(0), -1)
        means = self.fc1(data)
        return means




