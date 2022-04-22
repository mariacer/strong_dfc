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
# @title          :utils/math_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Utilities for mathematical computations
---------------------------------------

Functions required for mathematical computations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Define the activation functions.
ACTIVATION_FUNCTIONS = {
    'linear':    {'id': 0,
                  'fn':   lambda x: 1.*x,
                  'grad': lambda x: torch.ones_like(x)},
    'tanh':      {'id': 1,
                  'fn':   lambda x: torch.tanh(x),
                  'grad': lambda x: derivative_tanh(x)},
    'relu':      {'id': 2,
                  'fn':   lambda x: F.relu(x),
                  'grad': lambda x: derivative_relu(x)},
    'leakyrelu': {'id': 3,
                  'fn':   lambda x: F.leaky_relu(x, 0.2),
                  'grad': lambda x: derivative_leakyrelu(x)},
    'sigmoid':   {'id': 4,
                  'fn':   lambda x: torch.sigmoid(x),
                  'grad': lambda x: derivative_sigmoid(x)}
}

def get_activation_from_id(nl_id, grad=True):
    """From the activation id, return the function.

    Args:
        nl_id (int): The non-linearity id.
        grad (boolean): Whether to return to gradient function instead of the
            activation itself.

    Return:
        The nonlinearity function.
    """
    for key in ACTIVATION_FUNCTIONS.keys():
        if ACTIVATION_FUNCTIONS[key]['id'] == nl_id:
            return ACTIVATION_FUNCTIONS[key]['fn' if not grad else 'grad']

def compute_jacobian(x, y, structured_tensor=False,
                     retain_graph=False):
    """Compute the Jacobian matrix of output with respect to input.

    If input and/or output have more than one dimension, the Jacobian of the
    flattened output with respect to the flattened input is returned if
    `structured_tensor` is `False`. If `structured_tensor` is `True`, the 
    Jacobian is structured in dimensions `[y_shape, flattened_x_shape]`.
    Note that `y_shape` can contain multiple dimensions.
    
    Args:
        x (list or torch.Tensor): Input tensor or sequence of tensors with the
            parameters to which the Jacobian should be computed. Important:
            the `requires_grad` attribute of input needs to be `True` while
            computing output in the forward pass.
        y (torch.Tensor): Output tensor with the values of which the
            Jacobian is computed.
        structured_tensor (bool): A flag indicating if the Jacobian should be
            structured in a tensor of shape `[y_shape, flattened_x_shape]`
            instead of `[flattened_y_shape, flattened_x_shape]`.

    Returns:
        (torch.Tensor): 2D tensor containing the Jacobian of output with
            respect to input if `structured_tensor` is `False`.
            If `structured_tensor` is `True`, the Jacobian is structured in a
            tensor of shape `[y_shape, flattened_x_shape]`.
    """
    if isinstance(x, torch.Tensor):
        x = [x]

    # Create the empty Jacobian.
    output_flat = y.view(-1)
    numel_input = 0
    for input_tensor in x:
        numel_input += input_tensor.numel()
    jacobian = torch.Tensor(y.numel(), numel_input)

    # Compute the Jacobian.
    for i, output_elem in enumerate(output_flat):
        if i == output_flat.numel() - 1:
            gradients = torch.autograd.grad(output_elem, x,
                                            retain_graph=retain_graph,
                                            create_graph=False,
                                            only_inputs=True)
        else:
            gradients = torch.autograd.grad(output_elem, x,
                                            retain_graph=True,
                                            create_graph=False,
                                            only_inputs=True)
        jacobian_row = torch.cat([g.view(-1).detach() for g in gradients])
        jacobian[i, :] = jacobian_row

    if structured_tensor:
        shape = list(y.shape)
        shape.append(-1) 
        jacobian = jacobian.view(shape)

    return jacobian

def nullspace(A, tol=1e-12):
    """Compute the nullspace of a certain matrix.

    Args:
        A (torch.Tensor): A matrix.
        tol (float): The tolerance level for determining what the nullspace is.

    Returns:
        (torch.Tensor): A matrix with vectors in the nullspace.
    """
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S<tol))
    V_null = V[:, null_start:]

    return V_null


def nullspace_relative_norm(A, x, tol=1e-12):
    """Compute the ratio between the norm of components of `x` that are in the
    nullspace of `A` and the norm of `x`.

    Args:
        A (torch.Tensor): A matrix.
        x (torch.Tensor): A certain vector.
        tol (float): The tolerance level for determining what the nullspace is.

    Returns:
        (float): The ratio.
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    A_null = nullspace(A, tol=tol)
    x_null_coordinates = A_null.t().mm(x)
    ratio = x_null_coordinates.norm()/x.norm()

    return ratio

def derivative_sigmoid(x):
    """Compute the derivative of the sigmoid for a given input.

    Args:
        x (torch.Tensor): The input.

    Returns:
        (torch.Tensor): The derivative at the input.
    """
    return torch.mul(torch.sigmoid(x), 1. - torch.sigmoid(x))

def derivative_relu(x):
    """Compute the derivative of the relu for a given input.

    Args:
        x (torch.Tensor): The input.

    Returns:
        (torch.Tensor): The derivative at the input.
    """
    grad = torch.ones_like(x)
    grad[x < 0] = 0
    return grad

def derivative_leakyrelu(x):
    """Compute the derivative of the leaky relu for a given input.

    Args:
        x (torch.Tensor): The input.

    Returns:
        (torch.Tensor): The derivative at the input.
    """
    grad = torch.ones_like(x)
    grad[x < 0] = 0.2
    return grad

def derivative_tanh(x):
    """Compute the derivative of the tanh for a given input.

    Args:
        x (torch.Tensor): The input.

    Returns:
        (torch.Tensor): The derivative at the input.
    """
    return 1 - torch.tanh(x)**2

def outer(v1, v2):
    """Compute the outer product between two vectors.
    
    Simple wrapper to deal with several torch versions.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        (torch.Tensor): The outer product.
    """
    try:
        return torch.outer(v1, v2)
    except:
        return torch.ger(v1, v2)

def contains_nan(x, max_value=float('inf')):
    """Check whether a tensor contains a NaN or an infinity value.

    Args:
        x (torch.Tensor or list): The input.
        max_value (float): The highest acceptable value.

    Returns:
        (bool): Whether the tensor contains a NaN or infinity.
    """
    nb_nans = 0
    nb_infs = 0
    if isinstance(x, list):
        for el in x:
            nb_nans += torch.isnan(el).sum()
            nb_infs += (torch.abs(el) > float('inf')).sum()
    else:
        nb_nans = torch.isnan(x).sum()
        nb_infs = (torch.abs(x) > float('inf')).sum()
    
    return nb_nans > 0 or nb_infs > 0

def euclidean_dist(v1, v2, axis=None):
    """Compute the Euclidean distance between two vectors.

    If only 1D vectors, a scalar is returned. If a 2D or 3D matrix is fed,
    the first dimension is interpreted as time and vector/matrix distance
    is computed along it.

    Args:
        v1 (torch.Tensor): The first vector.
        v2 (torch.Tensor): The second vector.
        axis (int): The axis along which to compute the norm.

    Returns:
        The distance.
    """
    if axis is None:
        if len(v1.shape) == 1: 
            axis = 0        # normal 1D vectors
        elif len(v1.shape) == 2:
            axis = 1        # time x vector matrix, calculate along time
        elif len(v1.shape) == 3:
            axis = (1, 2)  # time x batch x vector matrix, calculate along time

    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        d = torch.norm(v1 - v2, dim=axis, p=2).detach()
    elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        d = np.linalg.norm(v1 - v2, axis=axis)
    else:
        raise ValueError('Invalid types {},{} for dist function'.format(
            type(v1), type(v2)))

    return d

def bool_to_indices(bool_vector):
    """Convert an array of boolean indices to integer indices.

    Args:
        bool_vector (torch.Tensor): A vector of booleans.

    Returns:
        (list): The list of indices that are ``True``.
    """
    indices_int = []

    for i in range(len(bool_vector)):
        if bool_vector[i]:
            indices_int.append(i)

    return indices_int

def get_jacobian_slice(network, layer_idx):
    """Returns the start and end indices of the columns in `J` that correspond
    to the network layer with index `layer_idx`.

    Args:
        network: The network.
        layer_idx (int): The index of the layer.

    Returns:
        (....): Tuple containing:

        - **neuron_index_start**: The index of the first neuron of the layer.
        - **neuron_index_end**: The index of the last neuron of the layer.
    """
    layer_output_dims = [l.weights.shape[0] for l in network.layers]
    neuron_index_start = sum(layer_output_dims[:layer_idx])
    neuron_index_end = sum(layer_output_dims[:(layer_idx + 1)])

    return neuron_index_start, neuron_index_end

def split_in_layers(network, layers_concat):
    r"""Split a Tensor containing the concatenated layer activations.

    Split a Tensor containing the concatenated layer activations for a
    minibatch into a list containing the activations of layer ``i`` at
    index ``i``.

    Args:
        network: The network.
        layers_concat (torch.Tensor): A tensor of dimension
            :math:`B \times \sum_{l=1}^L n_l` containing the concatenated
            layer activations..

    Returns:
        (list): A list containing values of ``layers_concat`` corresponding
            to the activations of layer ``i`` at index ``i``.
    """
    layer_output_dims = [l.weights.shape[0] for l in network.layers]
    start_output_limits = [sum(layer_output_dims[:i]) for i in
                           range(len(network.layers))]
    end_output_limits = [sum(layer_output_dims[:i + 1]) for i in
                         range(len(network.layers))]
    if len(layers_concat.shape) == 1:
        # to avoid errors when batch_size==1
        layers_concat = layers_concat.unsqueeze(0)

    return [layers_concat[:, start_output_limits[i]:end_output_limits[i]]
                             for i in range(len(network.layers))]

def flatten_list(unflattened_list):
    """Flatten list possibly containing lists within elements.

    Args:
        unflattened_list (list): The list to be flattened.

    Returns:
        (list): The flattened list.
    """
    flattened_list = []
    for li in unflattened_list:
        if isinstance(li, list):
            flattened_list.extend(flatten_list(li))
        else:
            flattened_list.append(li)

    return flattened_list

def compute_angle(A, B):
    """Compute the angle between two tensors of the same size. 

    The tensors will be flattened, after which the angle is computed.

    Args:
        A (torch.Tensor): First tensor.
        B (torch.Tensor): Second tensor.

    Returns:
        (float): The angle between the two tensors in degrees.
    """
    if contains_nan(A):
        warnings.warn('Tensor A contains NaNs, computing angle anyways.')
    if contains_nan(B):
        warnings.warn('Tensor B contains NaNs, computing angle anyways.')

    # Compute cosine angle.
    inner_product = torch.sum(A*B) # equal to inner product of flattened tensors
    cosine = inner_product/(torch.norm(A, p='fro')*torch.norm(B, p='fro'))
    if cosine > 1 and cosine < 1 + 1e-5:
        cosine = torch.Tensor([1.])

    # Convert to degrees.
    angle = 180 / np.pi * torch.acos(cosine)

    if contains_nan(angle):
        warnings.warn('Angle computation causes NaNs.')

    return angle

def vectorize_tensor_list(tensor_list):
    """ Vectorize all tensors in list.

    The tensors are all vectorized and concatenated in one single vector.

    Args:
        (list): The list of tensors.

    Returns:
        (torch.Tensor): The vectorized form.
    """

    return torch.cat([t.view(-1).detach() for t in tensor_list])

def cross_entropy(predictions, targets):
    """Home-made implementation of the cross-entropy.

    The mean or sum reduction is applied outside.

    Args:
        predictions (torch.Tensor): The predictions.
        targets (torch.Tensor): The targets.
        reduction (str): The type of reduction: `mean` or `sum`.

    Returns:
        (float): The loss for all items in the mini-batch.
    """
    # Apply the log softmax.
    logsoftmax = torch.log_softmax(predictions, dim=1)

    # Compute categorical cross-entropy, summing across classes.
    loss = - (targets * logsoftmax).sum(dim=1)

    return loss
        
def cross_entropy_fn(reduction='mean'):
    """Wrapper for the cross entropy function.

    We reimplement the cross-entropy so that we can use soft targets.
    For non one-hot-encodings or soft labels, we use Pytorch's native
    implementation.

    Args:
        reduction (str): The type of reduction: `mean` or `sum`.

    Returns:
        The loss function.
    """
    def cross_entropy_mean(predictions, targets):
        if len(targets.shape) == 1:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            return loss_fn(predictions, targets)
        else:
            return cross_entropy(predictions, targets).mean()

    def cross_entropy_sum(predictions, targets):
        if len(targets.shape) == 1:
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            return loss_fn(predictions, targets)
        else:
            return cross_entropy(predictions, targets).sum()

    if reduction == 'mean':
        return cross_entropy_mean
    elif reduction == 'sum':
        return cross_entropy_sum