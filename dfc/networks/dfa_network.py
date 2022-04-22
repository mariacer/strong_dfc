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
# @title          :networks/dfa_network.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a network for Direct Feedback Alingment
---------------------------------------------------------

A network that is prepared to be trained with DFA.
"""
import numpy as np
import torch
import torch.nn as nn

from networks.network_interface import NetworkInterface
from networks.dfa_layer import DFALayer

class DFANetwork(NetworkInterface):
    r"""Implementation of a network for Direct Feedback Alignment.

    Args:
        (....): See docstring of class
            :class:`network_interface.NetworkInterface`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return 'DFANetwork'

    @property
    def layer_class(self):
        """Define the layer type to be used."""
        return DFALayer

    def forward(self, x):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            x (torch.Tensor): The input to the network.

        Returns:
            The output of the network.
        """
        y = x
        grad_out = torch.empty((x.shape[0], self._n_out))

        for i, layer in enumerate(self.layers):
            is_last_layer = i+1 == len(self.layers)
            y = layer.forward(y, grad_out, is_last_layer=is_last_layer)

        return y

    @property
    def params(self):
        """Access parameters.

        Only feedforward weights are learned.

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
        feedback_weights = [layer.weights_backward.data for layer in \
            self.layers]

        state_dict = {'forward_weights': forward_weights,
                      'feedback_weights': feedback_weights}

        return state_dict

    def load_state_dict(self, state_dict):
        """Load a state into the network.

        This function sets the forward and backward weights.

        Args:
            state_dict (dict): The state with forward and backward weights.
        """
        for l, layer in enumerate(self.layers):
            layer.weights.data = state_dict['forward_weights'][l]
            layer.weights_backward.data = state_dict['feedback_weights'][l]