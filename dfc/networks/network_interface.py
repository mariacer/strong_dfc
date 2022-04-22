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
# @title          :networks/network_interface.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of an interface to be used for all network implementations
-------------------------------------------------------------------------

A simple network wrapper to be used as a blueprint for all other network
classes.
"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn

from networks.layer_interface import LayerInterface
from utils import math_utils as mutils

class NetworkInterface(nn.Module, ABC):
    r"""Implementation of an interface for networks.

    The last layer is always set to linear. For classification tasks, a softmax
    will be applied when computing the loss.

    Args:
        n_in (int): Number of inputs.
        n_hidden (list): A list of integers, each number denoting the size of a
            hidden layer. If ``None``, there is no hidden layer.
        n_out (int): Number of outputs.
        activation (str): The nonlinearity used in hidden layers.
            If ``None``, no nonlinearity will be applied.
        bias (bool): Whether layers may have bias terms.
        initialization (str): The type of initialization to be applied.
    """
    def __init__(self, n_in, n_hidden, n_out, activation='relu', bias=True,
                 initialization='orthogonal', **kwargs):
        super().__init__()

        self._n_in = n_in
        self._n_hidden = n_hidden
        self._n_out = n_out
        self._depth = len(n_hidden) + 1
        self._bias = bias
        self._activation = activation
        self._initialization = initialization
        self.create_layers(**kwargs)

    @property
    def n_in(self):
        """Getter for read-only attribute :attr:`n_in`."""
        return self._n_in
    
    @property
    def n_hidden(self):
        """Getter for read-only attribute :attr:`n_hidden`."""
        return self._n_hidden

    @property
    def n_out(self):
        """Getter for read-only attribute :attr:`n_out`."""
        return self._n_out
    
    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def activation(self):
        """Getter for read-only attribute :attr:`activation`."""
        return self._activation

    @property
    def initialization(self):
        """Getter for read-only attribute :attr:`initialization`."""
        return self._initialization

    @property
    def name(self):
        raise NotImplementedError('TODO implement function')
        return 'NetworkInterface'

    @property
    def layer_class(self):
        """Define the layer type to be used."""
        raise NotImplementedError('TODO implement function')
        return LayerInterface

    def zero_grad(self):
        """Initialize all the gradients of the network parameters to zero.

        This affects both the forward and the backward parameters.
        """
        for param in self.params:
            if isinstance(param, list):
                assert self._bias or len(param) == 1
                for p in param:
                    p.grad = torch.zeros_like(p)
            else:
                param.grad = torch.zeros_like(param)

    def get_max_grad(self, params=None):
        """Return the maximum gradient across the parameters.
        
        Args:
            params (list): The list of parameters across which to compute the
                maximum. If ``None`` is provided, it is extracted from the
                `params` attribute of the class.

        Returns:
            (float): The maximum gradient encountered.
        """
        if params is None:
            params = self.params

        gmax = 0
        for param in params:
            if self._bias:
                assert isinstance(param, list)
                for p in param:
                    if p.grad is not None and p.max() > gmax:
                        gmax = p.grad.max()
            else:
                if param.grad is not None and param.max() > gmax:
                    gmax = param.grad.max()
        return gmax

    def create_layers(self, **kwargs):
        """Create layers."""
        if self._n_hidden is None:
            n_all = [self._n_in, self._n_out]
        else:
            n_all = [self._n_in] + self._n_hidden + [self._n_out]
        self.layers = nn.ModuleList()

        ### Initialize the layers.
        for i in range(1, len(n_all)-1):
            layer = self.layer_class(n_all[i-1], n_all[i],
                last_layer_features=self._n_out, bias=self._bias,
                forward_activation=self._activation, requires_grad=True,
                initialization=self._initialization, **kwargs)
            self.layers.append(layer)
        # Output layer.
        output_layer = self.layer_class(n_all[-2], n_all[-1],
                last_layer_features=self._n_out, bias=self._bias,
                forward_activation='linear', requires_grad=True,
                initialization=self._initialization, **kwargs)
        self.layers.append(output_layer)

    def forward(self, x):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            x (torch.Tensor): The input to the network.

        Returns:
            The output of the network.
        """
        y = x

        for layer in self.layers:
            y = layer.forward(y)

        return y

    def backward(self, loss, targets=None):
        """Compute the backward pass.

        As this is simple backprop, no special computations are needed here,
        autograd does it all.

        Args:
            loss (float): The loss.
            targets (torch.Tensor): The dataset targets. This will usually be
                ignored, but will become useful for DFC with strong feedback.
        """
        loss.backward()

    @abstractmethod
    def params(self):
        """Access parameters.

        Returns:
            params (list): The list of parameters. It has length equal to the
                number of layers. If a bias is being used, each element of the
                list is a two-length list with first weights and then biases.
                For DFC networks, it will also contain the feedback weights, and
                thus will have longer length.
        """
        raise NotImplementedError('TODO implement function')

    @abstractmethod
    def forward_params(self):
        """Access parameters.

        Returns:
            params (list): The list of forward parameters.
        """
        raise NotImplementedError('TODO implement function')

    def clone_params(self, params_type='params'):
        """Clone the parameters.
        
        Args:
            params_type (str): The type of params to be returned. For DFC
                networks, this can be useful to distinguish between
                "forward_params" and "backard_params".

        Returns:
            (list): With the same structure as `self.params` but cloned values.
        """
        params_cloned = []
        params = getattr(self, params_type)
        for param in params:
            if isinstance(param, list):
                clone = [p.clone() for p in param]
            else:
                clone = [param.clone()]
            params_cloned.append(clone)
            
        return params_cloned

    def save_logs(self, writer, step, log_weights='forward', prefix=''):
        """Log the norm of the weights and the gradients.

        Args:
            writer: The tensorboard writer.
            step (int): The writer iteration.
            log_weights (True): Whether to log the forward weights.
            prefix (str): The naming prefix.
        """
        def log_param(param, param_name, i, step):
            """Log norm and gradient of one set of parameters."""
            # Log the weights.
            weights_norm = torch.norm(param)
            writer.add_scalar(tag='{}layer_{}/{}_norm'.format(prefix,
                                    int(i/2+1), param_name),
                              scalar_value=weights_norm,
                              global_step=step)
            
            # Log the norms.
            if param.grad is not None:
                gradients_norm = torch.norm(param.grad)
                writer.add_scalar(
                    tag='{}layer_{}/{}_gradients_norm'.format(prefix,
                                    int(i/2+1), param_name),
                    scalar_value=gradients_norm,
                    global_step=step)

        if log_weights == 'forward':
            for i, param in enumerate(self.params):
                log_param(param[0], 'weights', i, step)
                if len(param) == 2:
                    log_param(param[1], 'bias', i, step)

    def contains_nans(self, max_value=1000):
        """Check whether the network parameters contain NaNs or large values.

        Args:
            max_value (float): The maximum value above which parameters are
                considered to be diverging.

        Returns:
            (bool): Flag indicating whether the network contains a NaN.
                Also returns True if some parameters are above 1000, which
                indicates divergence.
        """
        for p in self.params:
            if mutils.contains_nan(p, max_value=max_value):
                return True
        return False

    def get_vectorized_parameter_updates(self, with_bias=True):
        """Get a vector with all the vectorized, concatenated parameter updates.

        Args:
            with_bias (bool): Whether to include biases, if they exist.

        Returns:
            (torch.Tensor): The vectorized form.
        """
        params = self.get_forward_parameter_list(with_bias)
        return torch.cat([p.grad.view(-1).detach() for p in params])

    def get_forward_parameter_list(self, with_bias=True):
        """Get a list of forward parameters with or without biases.

        Since `forward_params` might contain sublists, this transforms it into
        a flat list.

        Args:
            with_bias (boolean): Whether to include biases.

        Returns:
            (list): A flat list of parameters.
        """
        params = []
        for param in self.forward_params:
            if isinstance(param, list):
                params.append(param[0])
                if with_bias:
                    params.append(param[1])
            else:
                params.append(param)

        return params