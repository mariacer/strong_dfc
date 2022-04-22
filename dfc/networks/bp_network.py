#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany, Maria Cervera
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
# @title          :networks/bp_network.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a simple network that is trained with backpropagation
-----------------------------------------------------------------------

A simple network that is prepared to be trained with backprop.
"""
import numpy as np
import torch
import torch.nn as nn

from networks.network_interface import NetworkInterface
from networks.layer_interface import LayerInterface

class BPLayer(LayerInterface):
    """Implementation of a backpropagation layer.
    
    Args:
        (....): See docstring of class :class:`layer_interface.LayerInterface`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return 'BPLayer'
        
class BPNetwork(NetworkInterface):
    r"""Implementation of a Multi-Layer Perceptron (MLP) trained with backprop.

    This is a simple fully-connected network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    Args:
        (....): See docstring of class
            :class:`network_interface.NetworkInterface`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return 'BPNetwork'

    @property
    def layer_class(self):
        """Define the layer type to be used."""
        return BPLayer

    @property
    def params(self):
        """Access parameters.

        Returns:
            params (list): The list of parameters.
        """
        params = []
        for layer in self.layers:
            if not self._bias:
                try:
                    params.append([layer.weight])
                except:
                    params.append([layer.weights])
            else:
                try:
                    params.append([layer.weight, layer.bias])
                except:
                    params.append([layer.weights, layer.bias])

        return params

    @property
    def forward_params(self):
        """Access forward parameters.

        Returns:
            params (list): The list of parameters.
        """
        return self.params

    @property
    def state_dict(self):
        """A dictionary containing the current state of the network,
        incliding forward and backward weights.

        Returns:
            (dict): The forward and feedback weights.
        """
        forward_weights = [layer.weights.data for layer in self.layers]

        sd = {'forward_weights': forward_weights}

        return sd

    def load_state_dict(self, state_dict):
        """Load a state into the network.

        This function sets the forward and backward weights.

        Args:
            state_dict (dict): The state with forward and backward weights.
        """
        for l, layer in enumerate(self.layers):
            layer.weights.data = state_dict['forward_weights'][l]