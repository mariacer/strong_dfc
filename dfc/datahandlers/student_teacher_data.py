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
# @title          :datahandlers/student_teacher_data.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Teacher-based Dataset
---------------------

Implementation of dataloaders for the a teacher network-based dataset.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from networks.bp_network import BPNetwork

class RegressionDataset(Dataset):    
    """A teacher network-based regression dataset.

    In this setting, a teacher network is generated, and the dataset is obtained
    by feeding random inputs to this teacher network.

    Args:
        device: The cuda device where to place the dataset.
        n_in (int): The dimensionality of the inputs.
        n_out (int): The dimensionality of the outputs.
        n_hidden (list): The dimensionality of the hidden layers.
        num_data (int): The number of datapoints to generate.
        activation (str): The activation function to be used.
        double_precision (boolean): Whether precision of 64 floats is used.
        random_seed (int): The random seed for data generation.
    """
    def __init__(self, device, n_in=5, n_out=5, n_hidden=[100,100],
                 num_data=3000, activation='tanh', double_precision=True,
                 random_seed=42):

        # Generate the data.
        fixed_random_seed = np.random.RandomState(random_seed)
        data = fixed_random_seed.uniform(low=-1, high=1, size=(num_data, n_in))
        self.data = torch.tensor(data, dtype=torch.double if double_precision
                                             else torch.float)
        self.teacher = BPNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                            activation=activation, bias=True,
                            initialization='teacher')
        with torch.no_grad():
            self.targets = self.teacher.forward(self.data)

        # Some important properties.
        self._in_size = n_in
        self._out_size = n_out

        # Convert to double precision if required.
        if double_precision:
            self.data = self.data.double()
            self.targets = self.targets.double()
            
        # Move to device.
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """Overwrite getitem function to return both images and targets.

        Args:
            index (int): Index.

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self):
        return int(self.data.shape[0])