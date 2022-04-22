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
# @title          :datahandlers/mnist_auto_data.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
MNIST Autoencoder Dataset
-------------------------

Implementation of dataloaders for the MNIST autoencoding dataset.
"""
import torch
from torchvision.datasets import MNIST

class MNISTAutoData(MNIST):    
    """An instance of the class represents the MNIST autoencoding dataset.

    In this dataset, the class labels are replaced by the input images, as
    the goal is to reconstruct them.

    Args:
        root (str): Root directory of dataset 
        device: The cuda device where to place the dataset.
        double_precision (boolean): Whether precision of 64 floats is used.
        train (boolean): Whether to create the dataset from training data.
    """

    def __init__(self, root, device, double_precision=True, train=True, 
                 **kwargs):
        super().__init__(root, train, **kwargs)

        # Normalize the data.
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Some important properties.
        self._in_size = 784
        self._out_size = 784

        # Convert to double precision if required.
        if double_precision:
            self.data = self.data.double()
            
        # Move to device.
        self.data, self.targets = self.data.to(device), self.data.to(device).detach()

    def __getitem__(self, index):
        """Overwrite getitem function to return both images and targets.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.flatten()
        target = target.flatten()

        return img, target
