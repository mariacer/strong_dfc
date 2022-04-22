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
# @title          :networks/layer_interface.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a pytorch layer to be used within networks
------------------------------------------------------------

Layer module to be used within network classes.
"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils.math_utils import ACTIVATION_FUNCTIONS
from networks.credit_assignment_functions import non_linear_function

class LayerInterface(nn.Module, ABC):
    """Implementation of an abstract layer.

    Args:
        in_features (int): The number of pre-neurons.
        out_features (int): The number of post-neurons.
        last_layer_features (int): The number of neurons in the last layer.
            Only provided for direct feedback applications, else ``None``.
        bias (boolean): Whether the layer has a bias or not.
        requires_grad (boolean): Whether the parameters require a gradient.
        forward_activation (str): The forward activation to be used.
        initialization (str): The initialization to be used.
    """
    def __init__(self, in_features, out_features, last_layer_features=None,
                 bias=True, requires_grad=False, forward_activation='tanh',
                 initialization='orthogonal'):
        nn.Module.__init__(self)
        self._forward_activation = forward_activation
        self._requires_grad = requires_grad
        self._use_bias = bias
        self._in_features = in_features
        self._out_features = out_features
        self._initialization = initialization

        # Place-holder for storing activations.
        self._activations = None

        # Create and initialize layers.
        self.set_layer(in_features, out_features, use_bias=self._use_bias,
                       requires_grad=self._requires_grad)
        self.init_layer(self._weights, self._bias,
                        initialization=initialization)

    @property
    def name(self):
        raise NotImplementedError('TODO implement function')
        return 'LayerInterface'

    def set_layer(self, in_features, out_features, use_bias=True,
                  requires_grad=True):
        """Create the network parameters.

        Args:
            in_features (int): The number of pre-neurons.
            out_features (int): The number of post-neurons.
            use_bias (boolean): Whether a bias should be created.
            requires_grad (boolean): Whether the gradient should be computed.
        """
        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=requires_grad)
        if use_bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=requires_grad)
        else:
            self._bias = None

    def init_layer(self, weights, bias=None, initialization='xavier'):
        """Initialize the network parameters."""
        if initialization == 'orthogonal':
            out_features, in_features = weights.shape
            gain = np.sqrt(6. / (in_features + out_features))
            nn.init.orthogonal_(weights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(weights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(weights)
        elif initialization == 'teacher':
            nn.init.xavier_normal_(weights, gain=3.)
        elif initialization == 'ones':
            torch.nn.init.constant_(weights, 1.)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))
        if bias is not None and self._use_bias:
            nn.init.constant_(bias, 0)

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def forward_activation(self):
        """ Getter for read-only attribute :attr:`forward_activation`"""
        return self._forward_activation

    @property
    def use_bias(self):
        """ Getter for read-only attribute :attr:`use_bias`"""
        return self._use_bias

    def forward_activation_function(self, x):
        """Compute element-wise forward activation based on activation function.

        Args:
            x (torch.Tensor): The input.

        Returns:
            (torch.Tensor): The post-linearity activation.
        """
        if self.forward_activation in ACTIVATION_FUNCTIONS.keys():
            return ACTIVATION_FUNCTIONS[self.forward_activation]['fn'](x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def compute_vectorized_jacobian(self, x):
        """Compute the vectorized Jacobian of the forward activation function.

        Compute vectorized Jacobian evaluated at the value `x`. The vectorized
        Jacobian is the vector with the diagonal elements of the real Jacobian
        as it is a diagonal matrix for element-wise functions. As `x` is a
        minibatch, the output will also be a mini-batch of vectorized Jacobia
        (thus a matrix).
        
        Args:
            x (torch.Tensor): The linear activations for the current mini-batch.

        Returns:
            (torch.Tensor): The Jacobian.
        """
        if self.forward_activation in ACTIVATION_FUNCTIONS.keys():
            return ACTIVATION_FUNCTIONS[self.forward_activation]['grad'](x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def requires_grad(self):
        """Set `require_grad` attribute of the activations to `True`."""

        self._activations.requires_grad = True

    def forward(self, x):
        """Compute the output of the layer.

        This method applies first a linear mapping with the parameters
        ``weights`` and ``bias``, after which it applies the forward activation
        function.
        
        Args:
            x (torch.Tensor): Mini-batch of size `[B, in_features]` with input
                activations from the previous layer or input.

        Returns:
            The mini-batch of output activations of the layer.
        """
        ## Code without overwriting autograd's functions:
        # a = x.mm(self.weights.t())
        # if self.bias is not None:
        #     a += self.bias.unsqueeze(0).expand_as(a)
        # self.activations = self.forward_activation_function(a)

        ### Compute overwriting autograd's functions:
        self.activations = non_linear_function(x, self.weights, b=self.bias,
                                nonlinearity=self.forward_activation)

        return self.activations

    def compute_bp_update(self, loss, retain_graph=False):
        """Compute the error backpropagation update for the forward parameters.

        Args:
            loss (float): The network loss.
            retain_graph (bool): Whether the graph of the network should be
                retained after computing the gradients or jacobians. If the
                graph will not be used anymore for the current minibatch
                afterwards, `retain_graph` should be `False`.

        Returns:
            (torch.Tensor): The gradients.
        """
        grads = torch.autograd.grad(loss, self.get_forward_parameters,
                                    retain_graph=retain_graph)
        # else: DELETEME
        #     grads = torch.autograd.grad(loss, self.weights,
        #                                 retain_graph=retain_graph)

        return grads

    def compute_bp_activation_updates(self, loss, retain_graph=False,
                                      linear=False):
        """Compute the error backpropagation teaching signal for activations.

        Args:
            (....): See docstring of method :meth:`compute_bp_update`.
            linear (bool): Flag indicating whether the update should be
                computed for the linear activations instead of the nonlinear
                activations.

        Returns:
            (torch.Tensor): A tensor containing the BP updates for the layer
                activations for the current mini-batch.
        """
        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        grads = torch.autograd.grad(loss, activations,
                                    retain_graph=retain_graph)[0].detach()
        return grads

    def compute_nullspace_relative_norm(self, output_activation,
                                        retain_graph=False):
        """Compute the norm of the components of the weight gradients that are
        in the nullspace of the jacobian of the output with respect to weights,
        relative to the norm of the weight gradients.

        Args:
            output_activation (torch.Tensor): The outputs of the layer.
            (....): See docstring of method :meth:`compute_bp_update`.

        Returns:
            (torch.Tensor): The relative norm.
        """
        if output_activation.shape[0] > 1:
            return torch.Tensor([0])
        J = math_utils.compute_jacobian(self.weights, output_activation,
                                        structured_tensor=False,
                                        retain_graph=retain_graph)
        weights_update_flat = self.weights.grad.view(-1)
        relative_norm = math_utils.nullspace_relative_norm(J,
                                                           weights_update_flat)
        return relative_norm

    def save_logs(self, writer, step, name):
        """Save logs and plots of this layer.

        Args:
            writer (SummaryWriter): Summary writer.
            step (int): The global step used for the x-axis of the plots.
            name (str): The name of the layer.
        """
        # Save norm and gradient of the weights.
        forward_weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            forward_weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(tag='{}/forward_weights_gradients_norm'.\
                                format(name),
                              scalar_value=forward_weights_gradients_norm,
                              global_step=step)

        # Save norm and gradient of the biases.
        if self.bias is not None:
            forward_bias_norm = torch.norm(self.bias)
            writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                              scalar_value=forward_bias_norm,
                              global_step=step)
            if self.bias.grad is not None:
                forward_bias_gradients_norm = torch.norm(self.bias.grad)
                writer.add_scalar(tag='{}/forward_bias_gradients_norm'.\
                                    format(name),
                                  scalar_value=forward_bias_gradients_norm,
                                  global_step=step)

    def get_forward_parameters(self, with_bias=True):
        """Return a list containing the forward parameters.

        In previous versions this was also called `get_forward_parameter_list`.

        Args:
            with_bias (boolean): Whether the bias should be returned.

        Returns:
            (list): A list with the parameters. It has length of one if only
                weights are returned, and length two if biases are returned.
        """
        if self.bias is not None and with_bias:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def get_forward_gradients(self, with_bias=True):
        """Return a list containing the gradients of the forward parameters.

        Args:
            with_bias (boolean): Whether the bias should be returned.

        Returns:
            (list): A list with the gradients. It has length of one if only
                weights are returned, and length two if biases are returned.
        """
        if self.bias is not None and with_bias:
            return [self.weights.grad, self.bias.grad]
        else:
            return [self.weights.grad]

    def compute_bp_update(self, loss, retain_graph=False):
        """Compute the error backpropagation update for the forward
        parameters of this layer, based on the given loss.

        Args:
            loss (float): The loss.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, `retain_graph` should be `False`.
        """
        if self.use_bias:
            grads = torch.autograd.grad(loss, [self.weights, self.bias],
                                        retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(loss, self.weights,
                                        retain_graph=retain_graph)

        return grads