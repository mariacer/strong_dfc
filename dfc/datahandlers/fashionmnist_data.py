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
# @title          :datahandlers/fashionmnist_data.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Fashion MNIST Dataset
---------------------

Implementation of dataloaders for the Fashion MNIST dataset.
"""
import torch
from torchvision.datasets import FashionMNIST

class FashionMNISTData(FashionMNIST):    
    """An instance of the class represents the Fashion MNIST dataset.

    Args:
        root (str): Root directory of dataset 
        device: The cuda device where to place the dataset.
        double_precision (boolean): Whether precision of 64 floats is used.
        train (boolean): Whether to create the dataset from training data.
        target_class_value (float): The value of the correct class for the
            targets. If set smaller to one, can be used to set soft targets.
            Note that unless a value different than 1 is used, the labels are
            simply the integer values, not one-hot encodings.
    """

    def __init__(self, root, device, double_precision=True, train=True, 
                 target_class_value=1, **kwargs):
        super().__init__(root, train, **kwargs)

        # Normalize the data.
        self.data = self.data.unsqueeze(1).float().div(255)

        # Some important properties.
        self._in_size = 784
        self._out_size = 10

        # Convert to double precision if required.
        if double_precision:
            self.data = self.data.double()
            
        # Move to device.
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        
        self.num_classes = 10
        self.target_class_value = target_class_value

    def __getitem__(self, index):
        """Overwrite getitem function to return both images and targets.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.flatten()

        # Convert into soft targets if needed.
        if self.target_class_value != 1:
            non_target_class_value = (1 - self.target_class_value) / \
                                     (self.num_classes - 1)

            soft_target = non_target_class_value * torch.ones(self.num_classes,
                                                              device=img.device)
            soft_target[target] = self.target_class_value

            assert torch.abs(soft_target.sum() - 1) < 1e-5
            target = soft_target

        return img, target
