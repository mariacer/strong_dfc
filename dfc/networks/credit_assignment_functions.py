#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
# limitations under the License.#
# @title          :networks/credit_assignment_functions.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Adding custom functions to PyTorch's autograd
---------------------------------------------

The module :mod:`lib.backprop_functions` contains custom implementations of
neural network components (layers, activation functions, loss functions, ...),
that are compatible with PyTorch its autograd_ package.

A new functionality can be added to autograd_ by creating a subclass of class
:class:`torch.autograd.Function`. In particular, we have to implement the
:meth:`torch.autograd.Function.forward` method (which computes the output of a
differentiable function) and the :meth:`torch.autograd.Function.backward`
method (which computes the partial derivatives of the output of the implemented
:meth:`torch.autograd.Function.forward` method with respect to all input tensors
that are flagged to require gradients).

.. _autograd:
    https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
import torch
from torch.autograd import Function

import utils.math_utils as mutils

class NonlinearFunction(Function):
    r"""Implementation of a fully-connected layer with activation function.

    This class is a ``Function`` that behaves similar to PyTorch's class
    :class:`torch.nn.Linear`, but it has a different backward function that
    includes the non-linearity already. Sincethis class implements the interface
    :class:`torch.autograd.Function`, we can use it to specify a custom
    backpropagation behavior.

    Assuming column vectors: layer input :math:`\mathbf{a} \in \mathbb{R}^M`,
    bias vector :math:`\mathbf{b} \in \mathbb{R}^N` and a weight matrix
    :math:`W \in \mathbb{R}^{N \times M}`, this layer simply computes

    .. math::
        :label: eq-single-sample

        \mathbf{z} = \sigma(W \mathbf{a} + \mathbf{b})

    (or :math:`\mathbf{z} = \sigma W \mathbf{a})` if :math:`\mathbf{b}` is
    ``None``), where :math:`\sigma` is the nonlinearity..

    The mathematical operation described for single samples in eq.
    :eq:`eq-single-sample`, is stated for the case of mini-batches below

    .. math::
        :label: eq-mini-batch

        Z = \sigma (A W^T + \tilde{B})

    where :math:`Z \in \mathbb{R}^{B \times N}` is the output matrix.
    """
    @staticmethod
    def forward(ctx, A, W, nonlinearity, b=None):
        r"""Compute the output of a non-linear layer.

        This method implements eq. :eq:`eq-mini-batch`.

        Args:
            ctx: A context. Should be used to store activations which are needed
                in the backward pass.
            A: A mini-batch of input activations :math:`A`.
            W: The weight matrix :math:`W`.
            nonlinearity (str): The name of the nonlinearity to be used.
            b (optional): The bias vector :math:`\mathbf{b}`.

        Returns:
            The output activations :math:`Z` as defined by eq.
            :eq:`eq-mini-batch`.
        """
        # Solution inspired by:
        # https://pytorch.org/docs/master/notes/extending.html
        ### Compute linear part.
        Z_pre = A.mm(W.t())
        if b is not None:
            Z_pre += b.unsqueeze(0).expand_as(Z_pre)
        ctx.save_for_backward(A, W, b, Z_pre) # save pre-nonlinearity activation

        ### Compute non-linearity.
        Z = mutils.ACTIVATION_FUNCTIONS[nonlinearity]['fn'](Z_pre)
        # We need to store which nonlinearity we used, such that we can compute
        # the derivative in the backward pass. However, only constants can be
        # stored in the context, so we store the nonlinearity id.
        ctx.constant = mutils.ACTIVATION_FUNCTIONS[nonlinearity]['id']

        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        r"""Backpropagate the gradients of :math:`Z` through this layer.

        The matrix ``grad_Z``, which we denote by
        :math:`\delta_Z \in \mathbb{R}^{B \times N}`, contains the partial
        derivatives of the scalar loss function with respect to each element
        from the :meth:`forward` output matrix :math:`Z`.

        This method backpropagates the global error (encoded in
        :math:`\delta_Z`) to all input tensors of the :meth:`forward` method,
        essentially computing :math:`\delta_A`, :math:`\delta_W`,
        :math:`\delta_\mathbf{b}`.

        For the linear component, these partial derivatives can be computed as
        follows:

        .. math::

            \delta_A &= \delta_Z W \\
            \delta_W &= \delta_Z^T A \\
            \delta_\mathbf{b} &= \sum_{b=1}^B \delta_{Z_{b,:}}

        where :math:`\delta_{Z_{b,:}}` denotes the vector retrieved from the
        :math:`b`-th row of :math:`\delta_Z`.

        These need to be multiplied by the derivative of the post-nonlinearity
        with respect to its input.

        Args:
            ctx: See description of argument ``ctx`` of method :meth:`forward`.
            grad_Z: The backpropagated error :math:`\delta_Z`.

        Returns:
            (tuple): Tuple containing:

            - **grad_A**: The derivative of the loss with respect to the input
              activations, i.e., :math:`\delta_A`.
            - **grad_W**: The derivative of the loss with respect to the weight
              matrix, i.e., :math:`\delta_W`.
            - **grad_nonlinearity**: Which is always `None`.
            - **grad_b**: The derivative of the loss with respect to the bias
              vector, i.e., :math:`\delta_\mathbf{b}`; or ``None`` if ``b`` was
              passed as ``None`` to the :meth:`forward` method.

            .. note::
                Gradients for input tensors are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        A, W, b, Z_pre = ctx.saved_tensors

        grad_A = None
        grad_W = None
        grad_b = None
        
        # Compute the gradient of the output w.r.t. pre-nonlinearity activations
        grad_nonlinearity_fn = mutils.get_activation_from_id(ctx.constant,
                                                             grad=True)
        grad_Z_pre = torch.mul(grad_Z, grad_nonlinearity_fn(Z_pre))

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = grad_Z_pre.mm(W)
        if ctx.needs_input_grad[1]:
            grad_W = grad_Z_pre.t().mm(A)
        # We need to look at the third position in the inputs because we also
        # provide the nonlinearity name to the forward method in second
        # position.
        if b is not None and ctx.needs_input_grad[3]:
            grad_b = grad_Z_pre.sum(0)

        return grad_A, grad_W, None, grad_b


def non_linear_function(A, W, b=None, nonlinearity='linear'):
    """An alias for using class :class:`NonlinearFunction`.

    Args:
        (....): See docstring of method :meth:`NonlinearFunction.forward`.
        nonlinearity (str): The nonlinearity to be used.
    """
    # Note, `apply()` doesn't allow keyword arguments, which is why we build
    # this wrapper.
    if b is None:
        return NonlinearFunction.apply(A, W, nonlinearity)
    else:
        return NonlinearFunction.apply(A, W, nonlinearity, b)


class DFANonlinearFunction(Function):
    r"""Implementation of a fully-connected layer with activation function.

    This class is very similar to ``NonlinearFunction`` but it provides layers
    to be used with Direct Feedback Alignment, i.e. it projects gradients of
    the last layer directly to all upstream layers, while keeping the forward
    pass unchanged.
    """
    @staticmethod
    def forward(ctx, A, W, B, grad_out, nonlinearity, is_last_layer, b=None):
        r"""Compute the output of a non-linear layer.

        Same as in `NonlinearFunction.forward` except that here we also store
        in the context the direct feedback matrix.

        Args:
            (....): See docstring of method :meth:`NonlinearFunction.forward`.
            B (torch.Tensor): The feedback connection matrices.
            grad_out (torch.Tensor): The gradient of the loss with respect to
                the last layer activation. Required to be directly projected
                to all upstream layers.
            is_last_layer (boolean): Whether this is the last layer of the net.

        Returns:
            (....): See docstring of method :meth:`NonlinearFunction.forward`.
        """
        # Solution inspired by:
        # https://pytorch.org/docs/master/notes/extending.html
        ### Compute linear part.
        Z_pre = A.mm(W.t())
        if b is not None:
            Z_pre += b.unsqueeze(0).expand_as(Z_pre)
        ctx.save_for_backward(A, W, B, grad_out, b, Z_pre)

        ### Compute non-linearity.
        Z = mutils.ACTIVATION_FUNCTIONS[nonlinearity]['fn'](Z_pre)

        # We need to store which nonlinearity we used, such that we can compute
        # the derivative in the backward pass. However, only constants can be
        # stored in the context, so we store the nonlinearity id.
        # Note that the last layer is always linear.
        if is_last_layer:
            ctx.constant = -1
        else:
            ctx.constant = mutils.ACTIVATION_FUNCTIONS[nonlinearity]['id']

        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        r"""Directly project the output gradients to this layer.

        The matrix ``grad_Z``, which we denote by
        :math:`\delta_Z \in \mathbb{R}^{B \times N}`, contains the partial
        derivatives of the scalar loss function with respect to each element
        from the :meth:`forward` output matrix :math:`Z`. It is ignored for all
        layers, except for the last layer.

        For the linear component, the partial derivatives of the weights can be
        computed as follows:

        .. math::

            \delta_W &= B \delta_{Z_N}^T A \\

        Where :math:`B` is the feedback matrix that allows projecting from the
        output layer, and :math:`\delta_{Z_N}` is the gradient of the loss in
        the last layer.

        These need to be multiplied by the derivative of the post-nonlinearity
        with respect to its input.

        Args:
            ctx: See description of argument ``ctx`` of method :meth:`forward`.
            grad_Z: The backpropagated error :math:`\delta_Z`.

        Returns:
            (....): See docstring of method :meth:`NonlinearFunction.backward`.
        """
        A, W, B, grad_out, b, Z_pre = ctx.saved_tensors

        grad_A = None # We don't care about this gradient anymore.
        grad_W = None
        grad_b = None

        # Determine from the constant whether this is the last layer.
        is_last_layer = False
        if ctx.constant == -1:
            is_last_layer = True
            ctx.constant = 0 # overwrite with id of linear activation

        # The last layer behaves pretty much like in normal backprop whereas
        # the rest needs to ignore the provided gradient and use that of the
        # final layer.
        if is_last_layer:
            # Overwrite the referenced output gradient for later use in
            # upstream layers. It needs to be an in-place operation.
            grad_out.data.copy_(grad_Z)
        else:
            grad_Z = grad_out.to(B.device).mm(B)

        # Compute the gradient of the output w.r.t. pre-nonlinearity activations
        grad_nonlinearity_fn = mutils.get_activation_from_id(ctx.constant,
                                                             grad=True)
        grad_nonlinearity = grad_nonlinearity_fn(Z_pre)
        grad_Z_pre = torch.mul(grad_Z, grad_nonlinearity)
        if ctx.needs_input_grad[1]:
            grad_W = grad_Z.t().mm(A)
        if b is not None and ctx.needs_input_grad[-1]:
            grad_b = grad_Z.sum(0)

        return grad_A, grad_W, None, None, None, None, grad_b

def non_linear_dfa_function(A, W, B, grad_out, is_last_layer=False, b=None,
                            nonlinearity='linear'):
    """An alias for using class :class:`DFANonlinearFunction`.

    Args:
        (....): See docstring of method :meth:`NonlinearFunction.forward`.
        grad_out (torch.Tensor): The loss gradient of the last layer.
        is_last_layer (boolean): Whether this is the last layer.
    """
    # Note, `apply()` doesn't allow keyword arguments, which is why we build
    # this wrapper.
    if b is None:
        return DFANonlinearFunction.apply(A, W, B, grad_out, nonlinearity,
                                          is_last_layer)
    else:
        return DFANonlinearFunction.apply(A, W, B, grad_out, nonlinearity,
                                          is_last_layer, b)
