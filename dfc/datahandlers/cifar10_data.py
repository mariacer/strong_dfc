#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :datahandlers/cifar10_data.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
CIFAR-10 Dataset
----------------

Implementation of dataloaders for the CIFAR-10 dataset.
"""
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

class CIFAR10Data(CIFAR10):    
    """An instance of the class represents the CIFAR-10 dataset.

    Args:
        root (str): Root directory of dataset 
        device: The cuda device where to place the dataset.
        double_precision (boolean): Whether precision of 64 floats is used.
        train (boolean): Whether to create the dataset from training data.
    """

    def __init__(self, root, device, double_precision=True, train=True, 
                 **kwargs):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        super().__init__(root, train, transform=transform, **kwargs)

        # Get in the correct format.
        self.data = torch.transpose(torch.tensor(self.data), 1, 3).float()
        self.targets = torch.tensor(self.targets)

        # Some important properties.
        self._in_size = 3072
        self._out_size = 10

        # Convert to double precision if required.
        if double_precision:
            self.data = self.data.double()

        # Normalize the data.
        self.data = self.data.div(255).sub_(0.1307).div_(0.3081)
            
        # Move to device.
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """Overwrite getitem function to return both images and targets.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

