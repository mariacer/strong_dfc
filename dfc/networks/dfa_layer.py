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
# @title          :networks/dfa_layer.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a layer for Direct Feedback Alingment
-------------------------------------------------------

A layer that is prepared to be trained with DFA.
"""
import numpy as np
import torch
import torch.nn as nn

from networks.credit_assignment_functions import non_linear_dfa_function
from networks.layer_interface import LayerInterface

class DFALayer(LayerInterface):
    """Implementation of a Direct Feedback Alignment layer.
    
    Args:
        (....): See docstring of class :class:`layer_interface.LayerInterface`.
        last_layer_features (int): The size of the output layer.
        initialization_fb (str): The initialization to use for the feedback
            weights. If `None` is provided, the same initialization function
            as for the forward weights will be used.
    """
    def __init__(self, in_features, out_features, last_layer_features,
                 bias=True, requires_grad=False, forward_activation='tanh',
                 initialization='orthogonal', initialization_fb=None):
        super().__init__(in_features, out_features, bias=bias,
                         requires_grad=requires_grad,
                         forward_activation=forward_activation,
                         initialization=initialization)

        if initialization_fb is None:
            initialization_fb = initialization
        self._initialization_fb = initialization_fb
        self._last_features = last_layer_features

        # Create and initialize feedback weights.
        self.set_direct_feedback_layer(last_layer_features, out_features)
        self.init_layer(self._weights_backward,
                        initialization=initialization_fb)

    @property
    def weights_backward(self):
        """Getter for read-only attribute :attr:`_weights_backward`."""
        return self._weights_backward

    def set_direct_feedback_layer(self, last_features, out_features):
        """Create the network backward parameters.

        This layer connects the output layer to a hidden layer. No biases are
        used in direct feedback layers. These backward parameters have no
        gradient as they are fixed.

        Args:
            (....): See docstring of method
                :meth:`layer_interface.LayerInterface.set_layer`.
        """
        self._weights_backward = nn.Parameter(torch.Tensor(out_features,
                                                           last_features),
                                              requires_grad=False)

    def forward(self, x, grad_out, is_last_layer=False):
        """Compute the output of the layer.

        This method applies first a linear mapping with the parameters
        ``weights`` and ``bias``, after which it applies the forward activation
        function.
        
        Args:
            x (torch.Tensor): Mini-batch of size `[B, in_features]` with input
                activations from the previous layer or input.
            grad_out (torch.Tensor): A tensor that will reference the gradient
                of the output, such that it can then be overwritten during the
                gradient computation of the last layer, and used to be
                projected to earlier layers.
            is_last_layer (boolean): Whether this is the last layer.

        Returns:
            The mini-batch of output activations of the layer.
        """
        self.activations = non_linear_dfa_function(x, self.weights,
                                    self.weights_backward.t(), b=self.bias,
                                    grad_out=grad_out,
                                    is_last_layer=is_last_layer,
                                    nonlinearity=self.forward_activation)

        return self.activations

    @property
    def name(self):
        return 'DFALayer'
