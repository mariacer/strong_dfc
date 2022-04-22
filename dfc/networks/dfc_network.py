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
# @title          :networks/dfc_network.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a network for Deep Feedback Control
-----------------------------------------------------

A network that is prepared to be trained with DFC.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

from networks.network_interface import NetworkInterface
from networks.dfc_layer import DFCLayer
from utils import math_utils as mutils

class DFCNetwork(NetworkInterface):
    r"""Implementation of a network for Deep Feedback Control.

    Note that forward and feedback weights are learned in two different phases
    that we call the wake and sleep phases, respectively.

    It contains the following important functions:

    * ``forward``: compute the forward pass across all layers.
    * ``backward``: compute the backward phase with controller on, and compute
      the gradients of the forward weights.
    * ``compute_feedback_gradients``: compute the gradients of feedback weights.
    * ``controller``: a proportional integral controller.

    In this two-phase DFC setting, the following options exist for defining the
    target activations. For forward weight learning, the target outputs are
    either feedforward activations nudged towards lower loss (default) or set as
    the actual supervised targets (if the option ``strong_feedback`` is active).
    For feedback weight learning, the target outputs are either the feedforward
    activations (default) or the supervised targets (if the option
    ``strong_feedback`` is active).

    Args:
        (....): See docstring of class
            :class:`network_interface.NetworkInterface`.
        initialization_fb (str): The initialization for feedback weights.
        ndi (boolean): Whether to compute the non-dynamical inversion, i.e. the
            analytical solution at steady-state instead of simulating the
            dynamics.
        cont_updates (boolean): Whether the forward weights are updated at
            steady-state or not. Feedback weights are always computed
            continuously.
        sigma: Std of Gaussian noise to corrupt activations during controller
            dynamics.
        sigma_output: Same as `sigma` but for output layer.
        sigma_fb: Same as "sigma" but for the sleep phase.
        sigma_output_fb: Same as `sigma_fb` but for output layer.
        sigma_init (float): Std for assing Gaussian noise to feedback weight
            initialization for "weight_product" inits.
        dt_di (float): The timestep to be used in the differential equations.
        dt_di_fb (float): Same as "dt_di" but for the sleep phase.
        alpha_di (float): Leakage gain of the feedback controller.
        alpha_di_fb (float): Same as "alpha" but for the sleep phase.
        tmax_di (float): Maximum number of iterations (timesteps) performed in
            the dynamical inversion of targets.
        tmax_di_fb (float): Same as "tmax_di" but for the sleep phase.
        epsilon_di (float): Constant to check for convergence.
        k_p (float): The gain factor of the proportional control.
        k_p_fb (float): Same as "k_p" but for the sleep phase.
        time_constant_ratio (float): Ratio of the time constant of the voltage
            dynamics w.r.t. the controller dynamics.
        time_constant_ratio_fb (float): Same as "time_constant_ratio" but for
            the sleep phase.
        inst_transmission (bool): Whether to assume an instantaneous
            transmission between layers.
        inst_transmission_fb (bool): Same as "inst_transmission" but for the
            sleep phase.
        apical_time_constant (float): Time constant of the apical compartment.
        apical_time_constant_fb (float): Same as "apical_time_constant" but for
            the sleep phase.
        inst_system_dynamics (bool): Whether the dynamics of the somatic
            compartments, should be approximated by their instantaneous
            counterparts.
        inst_apical_dynamics (bool): Whether the dyamics of the apical
            compartiment should be instantneous.
        proactive_controller (bool): Whether to use the teaching signal of the
            next time step for simulating the controller dynamics. 
        noisy_dynamics (bool): Whether dynamics should be corrupted with noise
            with std "sigma" or "sigma_fb".
        target_stepsize (float): Step size for computing the output target based
            on the output gradient.
        tau_f (float): The time constant for filtering the dynamics and the
            control signal.
        tau_noise (float): The time constant to filter the noise.
        forward_requires_grad (bool): Whether the forward pass requires autograd
            to compute gradients.
        include_non_converged_samples (bool): Whether samples that have not
            converged should be excluded.
        compare_with_ndi (bool): Whether the dynamical inversion results should
            be compared to the analytical solution. Should only ever be active
            if `ndi` is `False`.
        low_pass_filter_u (bool): Whether the control signal should be low-pass
            filtered.
        low_pass_filter_noise (bool): Whether the noise should be low-pass
            filtered. Only relevant if `noisy_dynamics==True`.
        use_jacobian_as_fb (bool): Whether to use the Jacobian for the
            feedback weights.
        save_ndi_updates (bool): Whether angle with the analytical updates
            should be computed. Causes a minor increase in computational load.
        save_df (bool): Whether angles should be stored in a dataframe.
        strong_feedback (bool): Whether the feedback should be strong. In this
            case, the outputs are not simply nudged but clamped to the desired
            values. In this setting, the linearization becomes very inaccurate,
            so it does not make sense to use an analytical solution, so
            `ndi` should be `False`.
        compute_jacobian_at (str): How to compute the Jacobian.
        scaling_fb_updates (bool): Whether to scale the feedback updates
            differently for different layers.
        learning_rule (str): The type of learning rule to use for the forward
            weights.
    """
    def __init__(self, n_in, n_hidden, n_out, activation='relu', bias=True,
                 initialization='xavier_normal', cont_updates=False,
                 initialization_fb='weight_product', ndi=False, 
                 sigma=0.36, sigma_fb=0.01, sigma_output=0.36,
                 sigma_output_fb=0.1, sigma_init=1e-3,
                 dt_di=0.02, dt_di_fb=0.001, alpha_di=0.001, alpha_di_fb=0.5,
                 tmax_di=500, tmax_di_fb=10, k_p=2.0, k_p_fb=0., epsilon_di=0.3,
                 time_constant_ratio=0.2, time_constant_ratio_fb=0.005,
                 inst_transmission=False, inst_transmission_fb=False,
                 apical_time_constant=-1, apical_time_constant_fb=None, 
                 inst_system_dynamics=False, inst_apical_dynamics=False,
                 proactive_controller=False, noisy_dynamics=False,
                 target_stepsize=0.01, tau_f=0.9, tau_noise=0.8,
                 forward_requires_grad=False,
                 include_non_converged_samples=True, compare_with_ndi=False,
                 low_pass_filter_u=False, low_pass_filter_noise=False,
                 use_jacobian_as_fb=False, save_ndi_updates=False,
                 save_df=False, strong_feedback=False,
                 compute_jacobian_at='full_trajectory',
                 freeze_fb_weights=False, scaling_fb_updates=False,
                 learning_rule='nonlinear_difference'):
        super().__init__(n_in, n_hidden, n_out, activation=activation,
                         bias=bias, initialization=initialization,
                         initialization_fb=initialization_fb)

        self._input = None
        self._initialization_fb = initialization_fb
        self._ndi = ndi
        self._cont_updates = cont_updates
        self._epsilon_di = epsilon_di
        self._sigma = sigma
        self._sigma_output = sigma_output
        self._sigma_fb = sigma_fb
        self._sigma_output_fb = sigma_output_fb
        self._sigma_init = sigma_init
        self._dt_di = dt_di
        self._dt_di_fb = dt_di_fb
        self._alpha_di = alpha_di
        self._alpha_di_fb = alpha_di_fb
        self._tmax_di = tmax_di
        self._tmax_di_fb = tmax_di_fb
        self._k_p = k_p
        self._k_p_fb = k_p_fb
        self._time_constant_ratio = time_constant_ratio
        self._time_constant_ratio_fb = time_constant_ratio_fb
        self._inst_transmission = inst_transmission
        self._inst_transmission_fb = inst_transmission_fb
        self._apical_time_constant = apical_time_constant
        self._apical_time_constant_fb = apical_time_constant_fb
        self._inst_system_dynamics = inst_system_dynamics
        self._inst_apical_dynamics = inst_apical_dynamics
        self._proactive_controller = proactive_controller
        self._noisy_dynamics = noisy_dynamics
        self._target_stepsize = target_stepsize
        self._tau_f = tau_f
        self._tau_noise = tau_noise
        self._forward_requires_grad = forward_requires_grad
        self._include_non_converged_samples = include_non_converged_samples
        self._compare_with_ndi = compare_with_ndi
        self._low_pass_filter_u = low_pass_filter_u
        self._low_pass_filter_noise = low_pass_filter_noise 
        self._use_jacobian_as_fb = use_jacobian_as_fb
        self._strong_feedback = strong_feedback
        self._compute_jacobian_at = compute_jacobian_at
        self._freeze_fb_weights = freeze_fb_weights
        self._scaling_fb_updates = scaling_fb_updates
        self._learning_rule = learning_rule
        self.save_ndi_updates = save_ndi_updates
        self.save_df = save_df
        if compare_with_ndi:
            assert not ndi
            self.rel_dist_to_ndi = []

        if strong_feedback and self.ndi:
            raise ValueError('The analytical inversion is not applicable to '
                             'settings with strong feedback as the '
                             'linearization becomes inaccurate.')

        # This option is always initialized by default as MSE, and might be
        # overwritten to cross-entropy after the loss function has been defined.
        # This is used to compute the error based on the gradient of the loss.
        self._loss_function_name = 'mse'

        # Overwrite feedback weight initialization if necessary (up to output).
        if initialization_fb == 'weight_product':
            self.init_feedback_layers_weight_product()
        # Overwrite last feedback layer to be the identity.
        self.layers[-1].weights_backward = \
            torch.eye(self.layers[-1].weights_backward.shape[0]) 

        # If the homeostatic constant is -1, set it to the square of the
        # frobenius norm of the initialized feedback weights.
        self._homeostatic_const = float(torch.norm(self.full_Q, p='fro'))**2

        if not ndi:
            self.converged_samples_per_epoch = 0
            self.diverged_samples_per_epoch = 0
            self.not_converged_samples_per_epoch = 0

        if save_df:
            d = self._depth
            self.ndi_angles_network = pd.DataFrame(columns=[0])
            self.lu_angles_network = pd.DataFrame(columns=[0])
            self.ratio_ff_fb_network = pd.DataFrame(columns=[0])
            self.condition_gn = pd.DataFrame(columns=[0])
            self.condition_gn_init = pd.DataFrame(columns=[0])
            self.ndi_angles = pd.DataFrame(columns=[i for i in range(0, d)])
            self.bp_angles = pd.DataFrame(columns=[i for i in range(0, d)])
            self.lu_angles = pd.DataFrame(columns=[i for i in range(0, d)])
            self.ratio_ff_fb = pd.DataFrame(columns=[i for i in range(0, d)])

    @property
    def name(self):
        return 'DFCNetwork'

    @property
    def layer_class(self):
        """Define the layer type to be used."""
        return DFCLayer

    @property
    def r(self):
        """ Getter for attribute targets."""
        return self._r

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def ndi(self):
        """Getter for read-only attribute :attr:`ndi`."""
        return self._ndi

    @property
    def cont_updates(self):
        """ Getter for read-only attribute :attr:`cont_updates`"""
        return self._cont_updates

    @property
    def epsilon_di(self):
        """Getter for read-only attribute :attr:`epsilon_di`."""
        return self._epsilon_di

    @property
    def sigma(self):
        """Getter for read-only attribute :attr:`sigma`."""
        return self._sigma

    @property
    def sigma_fb(self):
        """Getter for read-only attribute :attr:`sigma_fb`."""
        return self._sigma_fb

    @property
    def sigma_init(self):
        """Getter for read-only attribute :attr:`_sigma_init`."""
        return self._sigma_init

    @property
    def sigma_output(self):
        """Getter for read-only attribute :attr:`sigma_output`."""
        return self._sigma_output

    @property
    def sigma_output_fb(self):
        """Getter for read-only attribute :attr:`sigma_output_fb`."""
        return self._sigma_output_fb

    @property
    def initialization_fb(self):
        """Getter for read-only attribute :attr:`initialization_fb`."""
        return self._initialization_fb

    @property
    def alpha_di(self):
        """Getter for read-only attribute :attr:`alpha_di`."""
        return self._alpha_di

    @property
    def alpha_di_fb(self):
        """Getter for read-only attribute :attr:`alpha_di_fb`."""
        return self._alpha_di_fb

    @property
    def dt_di(self):
        """Getter for read-only attribute :attr:`dt_di`."""
        return self._dt_di

    @property
    def dt_di_fb(self):
        """Getter for read-only attribute :attr:`dt_di_fb`."""
        return self._dt_di_fb

    @property
    def tmax_di(self):
        """Getter for read-only attribute :attr:`tmax_di`."""
        return self._tmax_di

    @property
    def tmax_di_fb(self):
        """Getter for read-only attribute :attr:`tmax_di_fb`."""
        return self._tmax_di_fb

    @property
    def k_p(self):
        """Getter for read-only attribute :attr:`k_p`"""
        return self._k_p

    @property
    def k_p_fb(self):
        """Getter for read-only attribute :attr:`k_p_fb`"""
        return self._k_p_fb

    @property
    def inst_system_dynamics(self):
        """Getter for read-only attribute :attr:`inst_system_dynamics`"""
        return self._inst_system_dynamics

    @property
    def inst_apical_dynamics(self):
        """Getter for read-only attribute :attr:`inst_apical_dynamics`"""
        return self._inst_apical_dynamics

    @property
    def noisy_dynamics(self):
        """Getter for read-only attribute :attr:`noisy_dynamics`"""
        return self._noisy_dynamics

    @property
    def inst_transmission(self):
        """ Getter for read-only attribute :attr:`inst_transmission`"""
        return self._inst_transmission

    @property
    def inst_transmission_fb(self):
        """ Getter for read-only attribute :attr:`inst_transmission_fb`"""
        return self._inst_transmission_fb

    @property
    def time_constant_ratio(self):
        """ Getter for read-only attribute :attr:`time_constant_ratio`"""
        return self._time_constant_ratio

    @property
    def time_constant_ratio_fb(self):
        """ Getter for read-only attribute :attr:`time_constant_ratio_fb`"""
        return self._time_constant_ratio_fb

    @property
    def apical_time_constant(self):
        """ Getter for read-only attribute :attr:`apical_time_constant`"""
        return self._apical_time_constant

    @property
    def apical_time_constant_fb(self):
        """ Getter for read-only attribute :attr:`apical_time_constant_fb`"""
        return self._apical_time_constant_fb

    @property
    def proactive_controller(self):
        """ Getter for read-only attribute :attr:`proactive_controller`"""
        return self._proactive_controller

    @property
    def target_stepsize(self):
        """ Getter for read-only attribute :attr:`target_stepsize`"""
        return self._target_stepsize
    
    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def include_non_converged_samples(self):
        """ Getter for read-only attribute 
        :attr:`include_non_converged_samples`"""
        return self._include_non_converged_samples

    @property
    def low_pass_filter_u(self):
        """ Getter for read-only attribute :attr:`low_pass_filter_u`"""
        return self._low_pass_filter_u

    @property
    def low_pass_filter_noise(self):
        """ Getter for read-only attribute :attr:`_low_pass_filter_noise`"""
        return self._low_pass_filter_noise

    @property
    def tau_f(self):
        """ Getter for read-only attribute :attr:`tau_f`"""
        return self._tau_f

    @property
    def tau_noise(self):
        """ Getter for read-only attribute :attr:`tau_noise`"""
        return self._tau_noise

    @property
    def use_jacobian_as_fb(self):
        """ Getter for read-only attribute :attr:`use_jacobian_as_fb`"""
        return self._use_jacobian_as_fb

    @property
    def strong_feedback(self):
        """ Getter for read-only attribute :attr:`strong_feedback`"""
        return self._strong_feedback

    @property

    def compute_jacobian_at(self):
        """ Getter for read-only attribute :attr:`compute_jacobian_at`"""
        return self._compute_jacobian_at

    @property
    def freeze_fb_weights(self):
        """ Getter for read-only attribute :attr:`freeze_fb_weights`"""
        return self._freeze_fb_weights

    @property
    def scaling_fb_updates(self):
        """ Getter for read-only attribute :attr:`scaling_fb_updates`"""
        return self._scaling_fb_updates

    @property
    def learning_rule(self):
        """ Getter for read-only attribute :attr:`learning_rule`"""
        return self._learning_rule

    @property
    def loss_function_name(self):
        """ Getter for read-only attribute :attr:`loss_function_name`"""
        return self._loss_function_name

    @loss_function_name.setter
    def loss_function_name(self, value):
        """Setter for loss_function_name"""
        self._loss_function_name = value

    @property
    def u(self):
        """ Getter for read-only attribute :attr:`u`"""
        return self._u

    @u.setter
    def u(self, value):
        """Setter for u"""
        self._u = value

    @property
    def full_Q(self):
        r""" Getter for matrix :math:`\bar{Q}` containing the concatenated
        feedback weights."""
        return torch.cat([l.weights_backward for l in self.layers], dim=0)

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
        for l, layer in enumerate(self._layers):
            layer.weights.data = state_dict['forward_weights'][l]
            layer.weights_backward.data = state_dict['feedback_weights'][l]

    def init_feedback_layers_weight_product(self):
        """Initialize the feedback weights to the inversion matrices.

        The feedback weights will be initializaed to the product of the forward
        weights (transposed) of subsequent layers.
        Note that this function can't be called at the level of inidividual
        layers since it needs information about the entire network.
        """
        for i in range(self.depth - 1):

            # Compute product of forward weights of subsequent layers.
            # Need to clone the tensor to not point to the same object.
            K = 1.*torch.transpose(self.layers[i+1].weights.data, 0, 1).clone()
            for j in range(i + 2, self.depth):
                K = torch.mm(K, torch.transpose(\
                             self.layers[j].weights.data, 0, 1))

            # Add noise if necessary.
            if self.sigma_init > 0.:
                K += torch.normal(mean=0., std=self.sigma_init, size=K.shape)

            self.layers[i].weights_backward = K

    def forward(self, x):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            x (torch.Tensor): The input to the network.

        Returns:
            The output of the network.
        """
        self.input = x
        y = x

        for i, layer in enumerate(self.layers):
            y = layer.forward(y)

        if y.requires_grad == False: # TODO check this
            y.requires_grad = True

        return y

    def backward(self, loss, targets=None, return_for_fb=False, verbose=True):
        """Run the feedback phase of the network.

        In this phase, the network is pushed to the output target by the
        controller. Compute the update of the forward weights of the network
        accordingly and save it in ``self.layers[i].weights.grad`` for each
        layer ``i``.
        
        Note: ``backward`` is implemented at the network-level, as it performs
        simultaneous inversion and control of all targets

        Args:
            loss (torch.Tensor): Mean output loss for current mini-batch.
            targets (torch.Tensor): The dataset targets. This will usually be
                ignored, as the targets will be taken to be the activations
                nudged towards lower loss, unless we use strong feedback.
            return_for_fb (boolean): Whether to return the control signal and
                the aplical voltages. This is useful when implementing DFC in
                a single phase, as these values will be needed to compute the
                feedback gradients.
            verbose (bool): Whether to display warnings.

        Returns:
            (torch.Tensor): (Optionally) the controller signal.
        """
        output_target = self.compute_output_target(loss, targets)

        # Get all pre- and post-nonlinearity activations.
        v_feedforward = [l.linear_activations for l in self.layers]
        r_feedforward = [l.activations for l in self.layers]

        # Compute the target activations.
        if self.ndi:
            u, v, r, r_out, delta_v = \
                self.non_dynamical_inversion(output_target)
        else:
            u, v, r, r_out, delta_v, \
                (u_time, v_fb_time, v_time, v_ff_time, r_time) = \
                    self.dynamical_inversion(output_target, verbose=verbose)
            if self._compare_with_ndi:
                # Compare analytical solution and compute relative distance.
                u_ndi, v_ndi, r_ndi, r_out_ndi, \
                    delta_v_ndi = self.non_dynamical_inversion(output_target)
                self.rel_dist_to_ndi.append(torch.mean(\
                    mutils.euclidean_dist(r_out, r_out_ndi)/
                    mutils.euclidean_dist(r_feedforward[-1], r_out_ndi)).\
                    detach().cpu().numpy())

        # Iterate across all layers.
        for i in range(self.depth):
            delta_v_i = delta_v[i]

            # Get the activations of the previous layer.
            if i == 0:
                r_previous = self.input
            else:
                r_previous = r[i - 1]

            # Compute the forward gradients.
            if self.cont_updates:
                if i == 0:
                    r_previous_time = self.input.unsqueeze(0).expand(
                        int(self.tmax_di), self.input.shape[0],
                        self.input.shape[1])
                else:
                    r_previous_time = r_time[i - 1]
                self.layers[i].compute_forward_gradients_continuous(v_time[i],
                        v_ff_time[i], r_previous_time,
                        learning_rule=self.learning_rule)
            else:
                self.layers[i].compute_forward_gradients(delta_v_i, r_previous,
                            learning_rule=self.learning_rule)

            # Store target values.
            self.layers[i].target = v[i]
            self.u = u

            if self.save_ndi_updates:
                # Compute and save the analytical updates.
                u_ndi, v_ndi, r_ndi, r_out_ndi, delta_v_ndi = \
                    self.non_dynamical_inversion(output_target)
                self.layers[i].compute_forward_gradients(delta_v_ndi[i],
                                             r_previous,
                                             saving_ndi_updates=True,
                                             learning_rule=self.learning_rule)
        self.u = u

        if return_for_fb and not self.ndi:
            return u_time, v_fb_time

    def compute_output_target(self, loss, targets=None):
        r"""Compute the output target.

        The target can be computed in one of two ways:

        * Nudged activations towards lower loss:

        .. math::

            \mathbf{r}_L^* = \mathbf{r}_L^- - \lambda \frac{\partial \
                \mathcal{L}}{\partial \mathbf{r}_L} \
                \bigg\rvert^T_{\mathbf{r}_L = \mathbf{r}_L^-}

        * As the actual targets (if ``strong_feedback==True``)

        .. math::

            \mathbf{r}_L^* = \mathbf{r}^\text{true}

        We assume the loss is averaged across the mini-batch, so here we need
        to multiply by the batch size to use the total loss over the mini-batch.

        Args:
            loss (torch.Tensor): Mean output loss for current mini-batch.
            targets (torch.Tensor): The dataset targets. This will usually be
                ignored, as the targets will be taken to be the activations
                nudged towards lower loss, unless we use strong feedback.

        Returns:
            (torch.Tensor): Mini-batch of output targets
        """
        if self.strong_feedback:
            # Return the targets.
            output_targets = targets
        else:
            # Return the nudged activations.
            target_lr = self.target_stepsize
            output_activations = self.layers[-1].activations

            # We multiply loss by batch size to have sum of losses.
            batch_size = output_activations.shape[0]

            # Compute gradient.
            gradient = torch.autograd.grad(loss*batch_size, output_activations,
                           retain_graph=self.forward_requires_grad)[0].detach()
            output_targets = output_activations - target_lr * gradient

        return output_targets

    @torch.no_grad()
    def dynamical_inversion(self, output_target, verbose=True):
        r"""Compute the dynamical (simulated) inversion of the targets.

        It does the inversion in real time, controlling all hidden layers
        simultaneously.

        This function calls ``self.controller()`` as a subroutine, which
        returns values for :math:`\mathbf{u}`, :math:`\mathbf{v}^\text{fb}`,
        :math:`\mathbf{v}`, :math:`\mathbf{v}^\text{ff}` and :math:`\mathbf{r}`
        for every simulated time step. The last values of these arrays are taken
        to represent the steady state. However, convergence is not guaranteed.
        If ``self.include_non_converged_samples`` is set to  ``False``,
        the values of batch elements that did not converge are set to
        their feedforward mode values, which includes :math:`\mathbf{u}` and
        :math:`\mathbf{v}^\text{fb}` being set to 0. 
        If ``self.include_non_converged_samples`` is set to ``True``, some of 
        the returned values with ``_ss`` suffix may in fact not represent the 
        steady state.

        Args:
            output_target (torch.Tensor): The output targets.
            verbose (bool): Whether to display warnings.
        
        Returns:
            (....): An ordered tuple containing:

            - **u_ss** (torch.Tensor): :math:`\mathbf{u}`, the final control
              input.
            - **v_ss** (list):
              :math:`\mathbf{v}_{ss} = \mathbf{v}^- + \
              \Delta_{\mathbf{v}}`
              The final voltage activations of the somatic
              compartments, split in a list that contains
              :math:`\mathbf{v}_{ss}` for each layer.
            - **r_ss** (list): :math:`\mathbf{r}_{ss} = \
              \phi(\mathbf{v}_{ss})`. The final firing rates of the
              neurons, split in a list that contains :math:`\mathbf{r}_{ss}`
              for each layer.
            - **r_out_ss** (torch.Tensor): The finaloutput activation
              of the network.
            - **delta_v_ss** (list): A list containing the final
              :math:`\Delta \mathbf{v}_i` for each layer.
            - **(u, v_fb, v, v_ff, r)** (tuple): A tuple with 5 elements.
              :math:`\mathbf{u}` represents a tensor of dimension
              :math:`t_{max}\times B \times n_L` containing the control
              input for each timestep.
              :math:`\mathbf{v}^\text{fb}`, :math:`\mathbf{v}`,
              :math:`\mathbf{v}^\text{ff}` each contain a list with at
              index ``i`` a ``torch.Tensor`` of dimension
              :math:`t_{max}\times B \times n_i`.
              :math:`\mathbf{r}` is a list with at index ``i`` a
              ``torch.Tensor`` of dimension :math:`t_{max}\times B \times n_i`
              containing the firing rates of layer ``i`` for each timestep.

        """
        batch_size = self.layers[0].activations.shape[0]

        # Get the post- and pre-nonlinearity activations.
        r_feedforward = [l.activations for l in self.layers]
        v_feedforward = [l.linear_activations for l in self.layers]

        # Compute the target activations.
        r, u, (v_fb, v_ff, v), sample_error = \
            self.controller(output_target, self.alpha_di, self.dt_di,
                            self.tmax_di,
                            k_p=self.k_p,
                            noisy_dynamics=self.noisy_dynamics,
                            inst_transmission=self.inst_transmission,
                            time_constant_ratio=self.time_constant_ratio,
                            apical_time_constant=self.apical_time_constant,
                            proactive_controller=self.proactive_controller,
                            sigma=self.sigma,
                            sigma_output=self.sigma_output)

        converged, diverged = self.check_convergence(r, r_feedforward,
                                                     output_target, u, 
                                                     sample_error, batch_size)

        # Select only samples that have converged.
        non_conv_idxs = converged == 0
        non_conv_idxs = mutils.bool_to_indices(non_conv_idxs)
        if not self.include_non_converged_samples:
            for i in range(self.depth):
                v[i][:, non_conv_idxs, :] = v_feedforward[i][non_conv_idxs, :]
                v_ff[i][:, non_conv_idxs, :] = v_feedforward[i][non_conv_idxs,:]
                v_fb[i][:, non_conv_idxs, :] = 0.
                r[i][:, non_conv_idxs, :] = r_feedforward[i][non_conv_idxs]
            u[:, non_conv_idxs, :] = 0.
            if verbose:
                warnings.warn('There are %s non-converged '%len(non_conv_idxs)+\
                              'samples that are discarded.')
        elif len(non_conv_idxs) > 0:
            if verbose:
                warnings.warn('There are %s non-converged '%len(non_conv_idxs)+\
                              'samples in the mini-batch.')

        # Get the steady-state target values (i.e. at last timestep).
        r_ss = [val[-1] for val in r]
        v_fb_ss = [val[-1] for val in v_fb]
        v_ff_ss = [val[-1] for val in v_ff]
        v_ss = [val[-1] for val in v]
        r_out_ss = r_ss[-1]
        u_ss = u[-1]

        # Compute the difference in somatic and basal voltages at steady-state.
        delta_v_ss = [v_ss[i] - v_ff_ss[i] for i in range(len(v_ss))]

        return u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r)
    
    @torch.no_grad()
    def non_dynamical_inversion(self, output_target, retain_graph=False):
        r"""Compute the analytical solution for the network activations in the
        feedback phase, when the controller pushes the network to reach the
        output target. The following formulas are used:

        .. math::

            \mathbf{u}_{ss} &= (JQ + \alpha I)^{-1} \delta_L \\
            \Delta \mathbf{v}_{ss} &= Q \mathbf{u}_{ss}

        with :math:`\mathbf{u}_{ss}` the control input at steady-state,
        :math:`\Delta \mathbf{v}_{ss}` the apical compartment voltage at
        steady-state, :math:`\delta_L` the difference between the output target
        and the output of the network at the feedforward sweep (without
        feedback). For the other symbols, refer to the paper.

        Args:
            output_target (torch.Tensor): The output target of the network.
            retain_graph (bool): Flag indicating whether the autograd graph
                should be retained for later use.

        Returns:
            (....): An ordered tuple containing:

            - **u_ndi** (torch.Tensor): :math:`\mathbf{u}_{ss}`, steady state
              control input.
            - **v_ndi** (list):
              :math:`\mathbf{v}_{ss} = \mathbf{v}^- + \
              \Delta \mathbf{v}_{ss}`
              The steady-state voltage activations of the somatic
              compartments, split in a list that contains
              :math:`\mathbf{v}_{ss}` for each layer.
            - **r_ndi** (list): :math:`\mathbf{r}_{ss} = \
              \phi(\mathbf{v}_{ss})`. The steady-state firing rates of the
              neurons, split in a list that contains :math:`\mathbf{r}_{ss}`
              for each layer.
            - **r_out_ndi** (torch.Tensor): The steady state output activation
              of the network.
            - **delta_v_ndi_split** (list): A list containing
              :math:`\Delta \mathbf{v}_{ss}` for each layer.
        """
        # Compute the Jacobian.
        Q = self.full_Q
        J = self.compute_full_jacobian()

        # Compute the error.
        deltaL = self.compute_error(output_target, self.layers[-1].activations)

        # Analytically compute the steady-state control signal.
        device = output_target.device
        if self.use_jacobian_as_fb:
            u_ndi = torch.solve(deltaL.unsqueeze(2), \
                        torch.matmul(J, J.transpose(1, 2)) + \
                        self.alpha_di * torch.eye(J.shape[1],device=device))\
                        [0].squeeze(-1)
            delta_v_ndi = torch.matmul(J.transpose(1, 2), \
                                u_ndi.unsqueeze(2)).squeeze(-1)
        else:
            u_ndi = torch.solve(deltaL.unsqueeze(2), torch.matmul(J, Q) + \
                        self.alpha_di * torch.eye(J.shape[1],device=device))\
                        [0].squeeze(-1)
            delta_v_ndi = torch.matmul(u_ndi, Q.t())
        delta_v_ndi_split = mutils.split_in_layers(self, delta_v_ndi)

        # Compute the targets across layers.
        r_previous = [self.input] 
        v_ndi = []
        for i in range(len(self.layers)):
            v_ndi.append(delta_v_ndi_split[i] + \
                torch.matmul(r_previous[i], self.layers[i].weights.t()))
            if self.layers[i].use_bias:
                v_ndi[i] += \
                    self.layers[i].bias.unsqueeze(0).expand_as(v_ndi[i])
            r_previous.append(\
                self.layers[i].forward_activation_function(v_ndi[i]))

        r_ndi = r_previous[1:]
        r_out_ndi = r_ndi[-1]

        return u_ndi, v_ndi, r_ndi, r_out_ndi, delta_v_ndi_split

    def compute_feedback_gradients(self, loss, targets, init=False):
        r"""Compute the gradients of the feedback weights for each layer.

        The updates are computed according to the following update rule:

        .. math::

            \frac{d Q_i}{dt} = - k \frac{1}{\sigma^2} \
                \mathbf{v}^{\text{fb}}_i(t) \mathbf{u}(t)^T - \beta Q_i

        where the output target is equal to the feedforward output of the
        network (i.e. without feedback) and where noise is applied to the
        network dynamics. Hence, the controller will try to 'counter' the
        noisy dynamics, such that the output is equal to the unnoisy output.

        The scaling :math:`k` is either 1 or, if the option
        ``scaling_fb_updates`` is active,
        :math:`(1 + \frac{\tau_v}{\tau_{\epsilon}})^{L-i}`.

        No inputs are required to this function since all activations are
        expected to be saved within the object.

        Args:
            loss (float): The loss.
            targets (torch.Tensor): The dataset targets. This will usually be
                ignored, as the targets will be taken to be the activations
                nudged towards lower loss, unless we use strong feedback.
            init (boolean): Indicates that this is a pre-train of the weights.
        """
        output_target = self.layers[-1].activations.data
        if self.strong_feedback:
            output_target = targets

        batch_size = output_target.shape[0]

        # Compute the controller signal.
        _, u, (v_fb, _, _), _ =  \
            self.controller(output_target=output_target,
                            alpha=self.alpha_di_fb,
                            dt=self.dt_di_fb,
                            tmax=self.tmax_di_fb,
                            k_p=self.k_p_fb,
                            noisy_dynamics=True,
                            inst_transmission=self.inst_transmission_fb,
                            time_constant_ratio=self.time_constant_ratio_fb,
                            apical_time_constant=self.apical_time_constant_fb,
                            proactive_controller=self.proactive_controller,
                            sigma=self.sigma_fb,
                            sigma_output=self.sigma_output_fb)

        # Compute gradient for each layer.
        u = u[1:, :, :] #  ignore the first timestep
        for i, layer in enumerate(self.layers):
            v_fb_i = v_fb[i][:-1, :, :]

            # compute a layerwise scaling for the feedback weights
            scaling = 1.
            if self.scaling_fb_updates:
                scaling = (1 + self.time_constant_ratio_fb / self.tau_noise) \
                     ** (len(self.layers) - i - 1)

            # get the amount of noise used
            sigma_i = self.sigma_fb
            if i == len(self.layers) - 1:
                sigma_i = self.sigma_output_fb

            layer.compute_feedback_gradients_continuous(v_fb_i, u,
                                                        sigma=sigma_i,
                                                        scaling=scaling)

    def compute_full_jacobian(self, linear=True, noisy_dynamics=False):
        r"""Compute the Jacobian of the network output.

        Compute the Jacobian of the network output (post-nonlinearity)
        with respect to either the concatenated pre-nonlinearity activations of
        all layers (including the output layer!)
        (i.e. ``self.layers[i].linear_activations``) if ``linear=True`` or the
        concatenated post-nonlinearity activations of all layers if
        ``linear=False`` (i.e. ``self.layers[i].activations``).

        If there is noise being used in the dynamics, then a low-passed version
        of the activations will be used to compute the Jacobian.

        Note that this implementation does not use autograd.

        Args:
            linear (bool): Flag indicating whether the Jacobian with respect to
                pre-nonlinearity activations (``linear=True``) should be taken
                or with respect to the post-nonlinearity activations
                (``linear=False``).
            noisy_dynamics (bool): Whether the dynamics are noisy.

        Returns:
            (torch.Tensor): A :math:`B \times n_L \times \sum_{l=1}^L n_l`
                dimensional tensor, with :math:`B` the minibatch size and
                :math:`n_l` the dimension of layer :math:`l`,
                containing the Jacobian of the network output w.r.t. the
                concatenated activations (pre or post-nonlinearity) of all
                layers, for each minibatch sample.
        """
        L = self.depth

        # Compute the derivatives w.r.t. activation function.
        if noisy_dynamics:
            vectorized_nonlinearity_derivative = \
                [l.compute_vectorized_jacobian(l.linear_activations_lp) \
                 for l in self.layers]
        else:
            vectorized_nonlinearity_derivative = \
                [l.compute_vectorized_jacobian(l.linear_activations) \
                 for l in self.layers]
        output_activation = self.layers[-1].activations
        
        batch_size = output_activation.shape[0]
        output_size = output_activation.shape[1]
        device = output_activation.device

        J = torch.empty(batch_size, output_size,
                        sum([l.weights.shape[0] for l in self.layers]),
                        device=device)

        # Compute Jacobian of the last layer.
        idx_start, idx_end = mutils.get_jacobian_slice(self, self.depth - 1)
        J[:, :, idx_start:idx_end] = \
            torch.eye(output_size, device=device).repeat(batch_size, 1, 1)

        if linear:
            J[:, :, idx_start:idx_end] = \
                    vectorized_nonlinearity_derivative[-1].view(\
                    batch_size, output_size, 1) * \
                    J[:, :, idx_start:idx_end]

        # Compute Jacobian of previous layers.
        for i in range(L - 1 - 1, 0 - 1, -1):
            # Get indices of the current layer.
            idx_start_1, idx_end_1 = mutils.get_jacobian_slice(self, i)
            # Get indices of the layer downstream to it.
            idx_start_2, idx_end_2 = mutils.get_jacobian_slice(self, i+1)
    
            if linear:
                J[:, :, idx_start_1:idx_end_1] = \
                        vectorized_nonlinearity_derivative[i].unsqueeze(1) * \
                        J[:, :, idx_start_2:idx_end_2].matmul(\
                        self.layers[i+1].weights)
            else:
                J[:, :, idx_start_1:idx_end_1] = \
                        (J[:, :, idx_start_2:idx_end_2] * \
                        vectorized_nonlinearity_derivative[i+1].unsqueeze(1)).\
                        matmul(self.layers[i+1].weights)

            del vectorized_nonlinearity_derivative[i]

        return J

    def controller(self, output_target, alpha, dt, tmax, k_p=0.,
                   noisy_dynamics=False, inst_transmission=False,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01):
        r"""Simulate the feedback control loop for several timesteps. 

        The following continuous time ODEs are simulated with time interval
        ``dt``. The following equation is used for the voltage:

        .. math::

            \frac{\tau_v}{\tau_u}\frac{d \mathbf{v}_i(t)}{dt} = \
                -\mathbf{v}_i(t) + W_i \mathbf{r}_{i-1}(t) + b_i + \
                Q_i \mathbf{u}(t)
            
        And the following for the control signal:

        .. math::

            \mathbf{u}(t) = \mathbf{u}^{\text{int}}(t) + k \mathbf{e}(t)

        .. math::

            \tau_u \frac{d \mathbf{u}^{\text{int}}(t)}{dt} = \mathbf{e}(t) - \
                \alpha \mathbf{u}^{\text{int}}(t)

        Note that we use a ratio :math:`\frac{\tau_v}{\tau_u}` instead of two
        separate time constants for :math:`\mathbf{v}` and :math:`\mathbf{u}`,
        as a scaling of both time constants can be absorbed in the simulation
        timestep ``dt``.
        IMPORTANT: ``time_constant_ratio`` should never be taken smaller than
        ``dt``, as then the forward Euler method will become unstable by
        default (the simulation steps will start to 'overshoot').

        If ``inst_transmission=False``, the forward Euler method is used to
        simulate the differential equation. If ``inst_transmission=True``, a
        slight modification is made to the forward Euler method, assuming that
        we have instant transmission from one layer to the next: the basal
        voltage of layer ``i`` at timestep ``t`` will already be based on the
        forward propagation of the somatic voltage of layer ``i-1`` at timestep
        ``t``, hence including the feedback of layer ``i-1`` at timestep ``t``.
        It is recommended to put ``inst_transmission=True`` when the
        ``time_constant_ratio`` is approaching ``dt``, as then we are
        approaching the limit of instantaneous system dynamics in the simulation
        where ``inst_transmission`` is always used (See below).

        If ``inst_system_dynamics=True``, we assume that the time constant of
        the system (i.e. the network) is much smaller than that of the
        controller and we approximate this by replacing the dynamical equations
        for :math:`\mathbf{v}_i` by their instantaneous equivalents:

        .. math::

            \mathbf{v}_i(t) = W_i \mathbf{r}_{i-1}(t) + b_i + Q_i \mathbf{u}(t)

        Note that ``inst_transmission`` will always be put on ``True`` 
        (overridden) in combination with ``inst_system_dynamics``.

        If ``proactive_controller=True``, the control input ``u[k+1]`` will be
        used to compute the apical voltages ``v^\text{fb}[k+1]``, instead of the
        control input ``u[k]``. This is a slight variation on the forward Euler
        method and corresponds to the conventional discretized control schemes.

        If ``noisy_dynamics=True``, noise is added to the apical compartment of
        the neurons. We now simulate the apical compartment with its own
        dynamics, as the feedback learning rule needs access to the noisy apical
        compartment. We use the following stochastic differential equation for
        the apical compartment:
        
        .. math::
        
            \tau_{\text{fb}} d \mathbf{v}_i^{\text{fb}}(t) = \
                (-\mathbf{v}_i^{\text{fb}}(t) + Q_i \mathbf{u}(t))dt + \sigma \
                \bm{\epsilon}_i(t)
        
        with :math:`\bm{\epsilon}` the Wiener process (Brownian motion) with
        covariance matrix :math:`I`.

        This is simulated with the Euler-Maruyama method:
        
        .. math::
        
            v_i^\text{fb}[k+1] = v_i^\text{fb}[k] + \Delta t / \tau_\text{fb} \
                (-v_i^\text{fb}[k] + Q_i u[k]) + \sigma / \sqrt{\Delta t / \
                \tau_\text{fb}} \Delta \beta

        with :math:`\Delta \beta` drawn from the zero-mean Gaussian distribution
        with covariance :math:`I`. The other dynamical equations in the system
        remain the same, except that :math:`Q_i \mathbf{u}` is replaced by
        :math:`\mathbf{v}_i^\text{fb}`:

        .. math::
        
            \tau_v \frac{d \mathbf{v}_i(t)}{dt} = -\mathbf{v}_i(t) + W_i \
                \mathbf{r}_{i-1}(t) + b_i + \mathbf{v}_i^\text{fb}

        One can opt for instantaneous apical compartment dynamics by putting
        its time constant :math:`\tau_\text{fb}` (``apical_time_constant``) equal
        to ``dt``. This is not encouraged for training the feedback weights, but
        can be used for simulating noisy system dynamics for training the
        forward weights, resulting in:

        .. math::

            \tau_v d \mathbf{v}_i(t) = (-\mathbf{v}_i(t) + W_i \
                \mathbf{r}_{i-1}(t) + b_i + Q_i \mathbf{u}(t) )dt + \
                \sigma \bm{\epsilon}_i(t)

        which can again be similarly discretized with the Euler-Maruyama method.

        Note that for training the feedback weights, it is recommended to put
        ``inst_transmission=True``, such that the noise of all layers can
        influence the output at the current timestep, instead of having to wait
        for a couple of timesteps, depending on the layer depth.

        Note that in the current implementation, we interpret that the noise is
        added in the apical compartment, and that the basal and somatic
        compartments are not noisy. At some point we might want to also add
        noise in the somatic and basal compartments for physical realism.

        Args:
            output_target (torch.Tensor): The output target
                :math:`\mathbf{r}_L^*` that is used by the controller to compute
                the control error :math:`\mathbf{e}(t)`.
            alpha (float): The leakage term of the controller.
            dt (float): The time interval used in the forward Euler method.
            tmax (int): The maximum number of timesteps.
            k_p (float): The positive gain parameter for the proportional part
                of the controller. If it is equal to zero (by default),
                no proportional control will be used, only integral control.
            noisy_dynamics (bool): Flag indicating whether noise should be
                added to the dynamics.
            inst_transmission (bool): Flag indicating whether the modified
                version of the forward Euler method should be used, where it is
                assumed that there is instant transmission between layers (but
                not necessarily instant voltage dynamics). See the docstring
                above for more information.
            time_constant_ratio (float): Ratio of the time constant of the
                voltage dynamics w.r.t. the controller dynamics.
            apical_time_constant (float): Time constant of the apical
                compartment. If ``-1``, we assume that the user does not want
                to model the apical compartment dynamics, but assumes instant
                transmission to the somatic compartment instead (i.e. apical
                time constant of zero).

        Returns:

            (....): Tuple containing:

            - **r** (list): A list with at index ``i`` a ``torch.Tensor``
                of dimension :math:`t_{max}\times B \times n_i` containing the
                firing rates of layer ``i`` for each timestep.
            - **u** (torch.Tensor): A tensor of dimension
                :math:`t_{max}\times B \times n_L` containing the control input
                for each timestep.
            - **(v_fb, v_ff, v)** (tuple): A tuple with 3 elements, each
                containing a list with at index ``i`` a ``torch.Tensor`` of
                dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical, basal or somatic compartments
                respectively.
            - **sample_error** (torch.Tensor): A tensor of dimension
                :math:`t_{max} \times B` containing the L2 norm of the error
                :math:`\mathbf{e}(t)` at each timestep.
        """
        if k_p < 0:
            raise ValueError('Only positive values for "k_p" are allowed')
        if self.inst_system_dynamics:
            inst_transmission = True

        if apical_time_constant == -1 or apical_time_constant == None:
            apical_time_constant = dt
        assert apical_time_constant > 0

        # Extract important variables and shapes.
        batch_size = output_target.shape[0]
        L = len(self.layers) 
        lod = [l.weights.shape[0] for l in self.layers] # layer out dims
        size_output = output_target.shape[1]
        tmax = int(tmax)
        device = output_target.device

        # Create empty containers for desired variables:
        # - v_fb: apical voltage (Ki u)
        # - v_ff: basal voltage (Wi h_target_i-1)
        # - v: somatic voltage
        # - r: v after non-linearlity
        # - u: control signal
        # - u_int: intermediate control signal if proportional part is active
        v_fb = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        v_ff = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        v = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        r = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        u = torch.zeros((tmax, batch_size, size_output),device=device)
        if k_p > 0:
            u_int = torch.zeros((tmax, batch_size, size_output),device=device)
        u_lp = None
        v_lp = None
        if self.low_pass_filter_u:
            u_lp = torch.zeros_like(u,device=device)
        if self.low_pass_filter_noise:
            noise_filtered = [torch.zeros((batch_size, l),device=device) for \
                              l in lod]
        if noisy_dynamics and self.use_jacobian_as_fb:
            v_lp = [torch.zeros((tmax, batch_size, l),device=device)\
                   for l in lod]
        sample_error = torch.ones((tmax, batch_size),device=device) * 10
        
        # Fill the values at the initial timestep.
        for i in range(L):
            v_ff[i][0, :] = self.layers[i].linear_activations
            v[i][0, :] = self.layers[i].linear_activations
            r[i][0, :] = self.layers[i].activations    
            if v_lp is not None:
                v_lp[i][0, :] = self.layers[i].linear_activations      
        sample_error[0] = self.compute_loss(output_target, r[-1][0, :])

        # Save initial zero targets for computation of Jacobian if needed.
        self._r = [r_l[:1, :] for r_l in r]

        # If hidden activations are linear, then J doen't depend on the samples
        if self.use_jacobian_as_fb and self.activation == 'linear':
            J = self.compute_full_jacobian(noisy_dynamics=noisy_dynamics)

        # Iterate over all the timesteps.
        for t in range(tmax - 1):

            # Compute the error.
            e = self.compute_error(output_target, r[-1][t])

            # If hidden activations are nonlinear, then J does depend on the
            # samples (derivative of their activations).
            if self.use_jacobian_as_fb and self.activation != 'linear':
                J = self.compute_full_jacobian(noisy_dynamics=noisy_dynamics)
            
            # Compute the control signal ``u``.
            if k_p > 0.:
                # Proportional and integral control.
                u_int[t + 1] = u_int[t] + dt * (e - alpha * u[t])
                u[t + 1] = u_int[t + 1] + k_p * e
            else:
                # Only integral control.
                u[t + 1] = u[t] + dt * (e - alpha * u[t])
            # Exponential low-pass filter if necessary.
            if self.low_pass_filter_u:
                # We need to keep track both of the unfiltered u and the
                # low-pass filtered u, as we might need the high-frequency parts
                # of u for the single-phase feedback weight updates.
                if t == 0:
                    # start the low-pass filtering at the same value of u,
                    # as otherwise it takes a long time to recover from zero
                    u_lp[t + 1] = u[t + 1]
                else:
                    u_lp[t + 1] = (dt / self.tau_f) * u[t + 1] + \
                                  (1 - (dt / self.tau_f)) * u_lp[t]

            def layer_iteration(i):
                """Compute the controlled activations of layer ``i``."""
                # Get the activities of previous layer.
                if i == 0:
                    r_previous = self.input
                else:
                    if inst_transmission:
                        r_previous = r[i - 1][t + 1]
                    else:
                        r_previous = r[i - 1][t]

                # Get basal voltage of current layer (based on ff input).
                a = r_previous.mm(self.layers[i].weights.t())
                if self.layers[i].bias is not None:
                    a += self.layers[i].bias.unsqueeze(0).expand_as(a)
                v_ff[i][t + 1, :] = a

                def get_control_signal(t, u_aux):
                    """Get the control signal Qu for the given timestep.

                    By default, this computes :math:`Qu` but in case the option
                    `use_jacobian_as_fb``is active, this computes :math:`Ju`.

                    Args:
                        t (int): The timestep.
                        u_aux (torch.Tensor): The control u to use. Can be
                            low-pass filtered or not, depending on
                            `low_pass_filter_u`.

                    Returns:
                        (torch.Tensor): The control signal.
                    """

                    if self.use_jacobian_as_fb:
                        batch_size = u_aux.shape[1]
                        n_out = u_aux.shape[2]

                        # Select the correct Jacobian block.
                        J_sq = J.view(batch_size * n_out, J.shape[-1])
                        Ji = mutils.split_in_layers(self, J_sq)[i]
                        Ji = Ji.view(batch_size, n_out, Ji.shape[-1])

                        return torch.matmul(u_aux[t].unsqueeze(1), Ji).squeeze()
                    else:
                        return torch.mm(u_aux[t], \
                                        self.layers[i].weights_backward.t())

                # Get the control signal.
                control_signal = get_control_signal(\
                                    t + 1 if proactive_controller else t,
                                    u_lp if self.low_pass_filter_u else u)
                assert control_signal.shape == v_fb[i][t, :].shape

                # Get apical voltage of current layer (based on fb input).
                if self.inst_apical_dynamics:
                    v_fb[i][t + 1, :] = control_signal
                else:
                    v_fb[i][t + 1, :] = v_fb[i][t, :] + dt / apical_time_constant *\
                                      (- v_fb[i][t, :] + control_signal)

                # Add noise to the apical voltage if necessary.
                if noisy_dynamics:
                    sigma_copy = sigma
                    if i == self.depth - 1:
                        sigma_copy = sigma_output
                    if self.low_pass_filter_noise:
                        # Warning: for very small dt, we might need to change
                        # the implementation for numerical stability and work
                        # with tau_noise * sqrt(dt) instead of 
                        # alpha_noise / sqrt(dt).
                        alpha_noise = dt / self.tau_noise
                        noise_filtered[i] = \
                                (alpha_noise / np.sqrt(dt)) * \
                                torch.randn_like(v_fb[i][t + 1, :],\
                                device=device)+\
                                (1 - alpha_noise) * noise_filtered[i]
                        v_fb[i][t + 1, :] +=  sigma_copy * noise_filtered[i]
                    else:   
                        v_fb[i][t + 1, :] +=  sigma_copy * np.sqrt(dt) / \
                            apical_time_constant * \
                            torch.randn_like(v_fb[i][t + 1, :],device=device)

                # Get somatic voltage as function of basal and apical voltages.
                if self.inst_system_dynamics:
                    v[i][t + 1, :] = v_fb[i][t + 1, :] + v_ff[i][t + 1, :]
                else: 
                    v[i][t + 1, :] = v[i][t, :] + dt / time_constant_ratio \
                                      * (v_fb[i][t + 1, :] + v_ff[i][t + 1, :] -
                                         v[i][t, :])

                # Compute the post-nonlinearity activations of the layer.
                r[i][t + 1, :] = \
                    self.layers[i].forward_activation_function(v[i][t + 1, :])

                # Update activations in layer objects to enable steady-state 
                # jacobian computation in `compute_full_jacobian()`
                if self.use_jacobian_as_fb:
                    if self.activation != 'linear':
                        self.layers[i].linear_activations = v[i][t + 1, :]
                        self.layers[i].activations = r[i][t + 1, :]
                    if noisy_dynamics:
                        alpha_r = dt / self.tau_f
                        v_lp[i][t + 1, :] = alpha_r * self.v[i][t + 1] + \
                                            (1 - alpha_r) * v_lp[i][t]
                        self.layers[i].linear_activations_lp = \
                                        [v_lp_l[:t + 1, :] for v_lp_l in v_lp]

            # Computed the controlled activations of all layers in current ts.
            if not inst_transmission:
                # Compute backwards to have already the influence of the
                # controller being propagated through network across time
                for i in range(L - 1, 0 - 1, -1):
                    layer_iteration(i)
            else:
                for i in range(L):
                    layer_iteration(i)

            # Compute the loss.
            sample_error[t + 1] = self.compute_loss(output_target,
                                                    r[-1][t + 1, :])

            # Save targets for computation of Jacobian if needed, only up to
            # the current timestep.
            self._r = [r_l[:t + 1, :] for r_l in r]

        # Store the control signal.
        if noisy_dynamics:
            # With noisy dynamics, the last value of u will be noisy, and we
            # should average over u to cancel out the noise. I assume that in
            # the last quarter of the simulation, u has converged, so we can
            # average over that interval.
            interval_length = int(tmax / 4)
            self.u = torch.sum(u[-interval_length:-1,:,:], dim=0)\
                    /float(interval_length)
        else:
            self.u = u[-1]

        return r, u, (v_fb, v_ff, v), sample_error

    def check_convergence(self, r, r_feedforward, output_target, u,
                          sample_error, batch_size):
        """Check whether the dynamics of the network have converged.

        This function computes whether individual samples have converged to a
        small output error. Like this, the ones that have not converged can be
        exclulded from the mini-batch update.

        Args:
            r (torch.Tensor): The target activations across layers.
            r_feedforward (torch.Tensor): The forward activations across layers.
            output_target (torch.Tensor): The output target.
            u (torch.Tensor): The control signal.
            sample_error (torch.Tensor): The L2 norm of the error :math:`e(t)`
                at each timestep, computed by the controller.
            batch_size (int): The batch-size.

        Returns:
            (....): Tuple containing:

            - **converged**: List indicating if individual samples converged.
            - **diverged**: List indicating if individual samples diverged.
        """
        # Define the thresholds for convergence and divergence.
        threshold_convergence = 1e-5
        threshold_divergence = 1

        # Get the target activations in the last layer and timestep.
        r = r[-1][-1]
        u = u[-1]

        # Compute the loss.
        diff = self.compute_loss((output_target - self.alpha_di * u), r)
        norm = torch.norm(r_feedforward[-1], dim=1).detach()

        # Determine whether individual samples converged or diverged.
        converged = ((diff / norm) < threshold_convergence) * \
                    (sample_error[-1] < self.epsilon_di ** 2 * sample_error[0])
        diverged = (diff / norm) > threshold_divergence

        # Update the count of converged/diverged samples per epoch.
        self.converged_samples_per_epoch += \
            sum(converged).detach().cpu().numpy()
        self.diverged_samples_per_epoch += sum(diverged).detach().cpu().numpy()
        self.not_converged_samples_per_epoch += \
            (batch_size - sum(converged) - sum(diverged)).detach().cpu().numpy()

        return converged, diverged

    def compute_error(self, output_target, r):
        r"""Compute the error :math:`\mathbf{e}(t)` in the predictions.

        By default this error is computed as in the DFC paper according to:

        .. math::

            \mathbf{e}(t) = \mathbf{r}_L^* - \mathbf{r}_L(t)

        For a mean-squared error (MSE) loss
        :math:`\mathcal{L} = \frac{1}{2}\lVert {\mathbf{r}_L^* - \
        \mathbf{r}_L(t)} \rVert_{2}^{2}`, this can be seen as the gradient of
        the loss with respect to the output activations :math:`\mathbf{r}_L(t)`.
        This notion can be generalized to other losses, and we can instead
        write the error as:

        .. math::

            \mathbf{e}(t) = - \frac{\partial \mathcal{L}}{\partial \
                \mathbf{r}_L} \biggr\rvert^T_{\mathbf{r}_L=\mathbf{r}_L(t)}

        In this function we hard-code the solution of this equation for the
        MSE loss mentioned above as well as for the cross-entropy loss. Which
        one of these is used will be determined by the attribute
        ``loss_function_name``, which by default is the MSE loss.

        So for cross-entropy loss, we return the following error:

        .. math::

            \mathbf{e}(t) = \mathbf{r}_L^* - \text{softmax}(\mathbf{r}_L(t))

        Args:
            output_target (torch.Tensor): The desired output
                :math:`\mathbf{r}_L^*`.
            r (torch.Tensor): The current output :math:`\mathbf{r}_L(t)`.

        Returns:
            (torch.Tensor): The error :math:`\mathbf{e}(t)`.
        """
        assert output_target.shape == r.shape
        if self.loss_function_name == 'mse':
            return output_target - r
        elif self.loss_function_name == 'cross_entropy':
            return output_target - torch.softmax(r, dim=1)
        else:
            raise ValueError('Loss function %s ' % self.loss_function_name + \
                             'not recognized.')

    def compute_loss(self, output_target, r, axis=1):
        r"""Compute the loss in the predictions.

        This function is mostly used to check for convergence.
        By default this error is computed as in the DFC paper for each sample
        according to:

        .. math::

            \mathcal{L} = \frac{1}{2}\ \lVert {\mathbf{r}_L^* - \
                \mathbf{r}_L(t)} \rVert_{2}^{2}

        However, if ``loss_function_name==cross_entropy`` we compute the
        following:

        .. math::

            \mathcal{L} = - (\mathbf{r}_L^* * \log \
                \text{softmax}(\mathbf{r}_L(t))

        Args:
            output_target (torch.Tensor): The desired output
                :math:`\mathbf{r}_L^*`.
            r (torch.Tensor): The current output :math:`\mathbf{r}_L(t)`.
            axis (int): The axis across which to compute the norm.

        Returns:
            (torch.Tensor): The list of loss values in the mini-batch.
        """
        assert output_target.shape == r.shape

        if self.loss_function_name == 'mse':
            return torch.norm(output_target - r, dim=axis, p=2).detach()
        elif self.loss_function_name == 'cross_entropy':
            return mutils.cross_entropy(r, output_target).detach()
        else:
            raise ValueError('Loss function %s ' % self.loss_function_name + \
                             'not recognized.')

    @property
    def forward_params(self):
        """Access forward parameters.

        Returns:
            params (list): The list of forward parameters.
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
    def feedback_params(self):
        """Access feedback parameters.

        Returns:
            params (list): The list of feedback parameters.
        """
        params = []
        for layer in self.layers:
            params.append([layer.weights_backward])

        return params

    @property
    def params(self):
        """Access all parameters.

        Returns:
            params (list): The list of all parameters.
        """
        params = self.forward_params
        params.extend(self.feedback_params)

        return params

    def get_max_grad(self, params_type='both'):  
        """Return the maximum gradient across the parameters.

        Args:
            params_type (str): Whether to compute across the entire set of
                parameters or only forward or only feedback.

        Returns:
            (float): The maximum gradient encountered.
        """
        if params_type == 'both':
            return super().get_max_grad(params=self.params)
        elif params_type == 'forward':
            return super().get_max_grad(params=self.forward_params)
        elif params_type == 'feedback':
            return super().get_max_grad(params=self.feedback_params)

    def save_logs(self, writer, step, log_weights='both', prefix=''):
        """Log the norm of the weights and the gradients.

        Args:
            writer: The tensorboard writer.
            step (int): The writer iteration.
            log_weights (True): Which weights to log.
            prefix (str): The naming prefix.
        """
        def log_param(param, param_name, i, step, feedback=False):
            """Log norm and gradient of one set of parameters."""
            # Log the weights.
            weights = 'feedback' if feedback else 'forward'
            weights_norm = torch.norm(param)
            writer.add_scalar(tag='{}layer_{}/{}_{}_norm'.format(prefix,
                                    int(i/2+1), param_name, weights),
                              scalar_value=weights_norm,
                              global_step=step)
            
            # Log the norms.
            if param.grad is not None:
                gradients_norm = torch.norm(param.grad)
                writer.add_scalar(
                    tag='{}layer_{}/{}_{}_gradients_norm'.format(prefix,
                                int(i/2+1), param_name, weights),
                    scalar_value=gradients_norm,
                    global_step=step)

        if log_weights in ['both', 'forward']:
            for i, param in enumerate(self.forward_params):
                log_param(param[0], 'weights', i, step)
                if len(param) == 2:
                    log_param(param[1], 'bias', i, step)

        if log_weights in ['both', 'feedback']:
            for i, param in enumerate(self.feedback_params):
                log_param(param[0], 'weights', i, step, feedback=True)
                if len(param) == 2:
                    log_param(param[1], 'bias', i, step, feedback=True)

    def to(self, device):
        """Override `to` method to also move the backward weights."""
        super().to(device)
        for i in range(self.depth):
            self.layers[i].weights_backward = \
                self.layers[i].weights_backward.to(device)

    def set_grads_to_bp(self, loss, retain_graph=False):
        """Set the gradients to correspond to those obtained with backprop.

        This function replaces the ``grad`` attributes of the forward weights.

        Args:
            loss (float): The current loss.
            retain_graph (boolean): Whether autograd graph should be retained.
        """
        # Autograd requires a flat list of parameters.
        flattened_params = mutils.flatten_list(self.forward_params)
        bp_grad = torch.autograd.grad(loss, flattened_params,
                                      retain_graph=retain_graph)
        unflattened_grad = self.unflatten_params(bp_grad)

        for i, param in enumerate(self.forward_params):
            if isinstance(param, list):
                for p, bpg in zip(param, unflattened_grad[i]):
                    assert p.shape == bpg.shape
                    p.grad = bpg.clone()
            else:
                assert param.shape == unflattened_grad[i].shape
                param.grad = unflattened_grad[i].clone()

    def unflatten_params(self, params, params_type='forward_params'):
        """Unflatten the parameters.

        This function assumes a certain structure in the parameters to unflatten
        a list into the same structure as `self.forward_params` or
        `self.feedback_params`.

        Args:
            params (list): The flat list.
            params_type (str): The type of params we are dealing with.

        Returns:
            (list): The unflattened list.
        """
        if isinstance(params, tuple):
            params = list(params)

        unflattened_params = []
        for i in range(self.depth):
            weights = params.pop(0)
            if self._bias:
                bias = params.pop(0)
                unflattened_params.append([weights, bias])
            else:
                unflattened_params.append([weights])

        # Sanity checks. REMOVE
        params_to_match = getattr(self, params_type)
        assert len(unflattened_params) == len(params_to_match)
        for unf_params, params in zip(unflattened_params, params_to_match):
            if isinstance(unf_params, list):
                assert isinstance(params, list)
                for a, b in zip(unf_params, params):
                    assert a.shape == b.shape
            else:
                assert unf_params.shape == params.shape

        return unflattened_params

    @property
    def forward_params_grad(self):
        """Return a structure identical to forward params but with gradients.

        Returns:
            (list): The gradients.
        """
        grads = []
        for params in self.forward_params:
            if isinstance(params, list):
                assert self._bias
                grads.append([params[0].grad, params[1].grad])    
            else:
                grads.append([params.grad])

        return grads
    
    def save_ndi_angles(self, writer, step, save_dataframe=True,
                        save_tensorboard=True):
        """Save angle between dynamical and analytical inversion results.

        The analytical results have been stored during training in
        `self.layers[i].ndi_update_weights`.

        Save the angle in the tensorboard writer (if ``save_tensorboard=True``)
        and in the corresponding dataframe (if ``save_dataframe=True``).

        Args:
            writer: Tensorboard writer.
            step (int): The number of forward training mini-batches.
            save_dataframe (bool): Flag indicating whether a dataframe of the
                angles should be saved in the network object.
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard.
        """
        ndi_param_updates = []
        net_param_updates = mutils.flatten_list(self.forward_params_grad)

        for i in range(self.depth):
            parameter_update = self.layers[i].get_forward_gradients()
            weights_angle = mutils.compute_angle(\
                    self.layers[i].ndi_updates_weights, parameter_update[0])
            ndi_param_updates.append(self.layers[i].ndi_updates_weights)
            if self._bias:
                bias_angle = mutils.compute_angle(\
                        self.layers[i].ndi_updates_bias, parameter_update[1])
                ndi_param_updates.append(self.layers[i].ndi_updates_bias)

            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(tag='{}/weight_ndi_angle'.format(name),
                                  scalar_value=weights_angle,
                                  global_step=step)
                if self._bias:
                    writer.add_scalar(tag='{}/bias_ndi_angle'.format(name),
                                      scalar_value=bias_angle,
                                      global_step=step)
            if save_dataframe:
                self.ndi_angles.at[step, i] = weights_angle.item()

        # Compute the total angle between the entire updates across all layers.
        total_angle = mutils.compute_angle(\
                            mutils.vectorize_tensor_list(ndi_param_updates),
                            mutils.vectorize_tensor_list(net_param_updates))
        if save_tensorboard:
            writer.add_scalar(tag='total_alignment/ndi_angle',
                              scalar_value=total_angle, global_step=step)
        if save_dataframe:
            self.ndi_angles_network.at[step, 0] = total_angle.item()


    def save_bp_angles(self, writer, step, loss, retain_graph=False,
                       save_tensorboard=True, save_dataframe=True):
        """Save the angles of the current forward parameter updates
        with the backprop update.

        Save the angle in the tensorboard writer (if ``save_tensorboard=True``)
        and in the corresponding dataframe (if ``save_dataframe=True``).

        Args:
            (....): See docstring of method :meth:`save_ndi_angles`.
            loss (torch.Tensor): Output loss of the network.
            retain_graph (bool): Flag indicating whether the graph of the
                network should be retained after computing the gradients or
                Jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, `retain_graph` should be `False`.
        """
        layer_indices = range(len(self.layers))

        for i in layer_indices:
            retain_graph_flag = retain_graph
            if i != layer_indices[-1]:
                retain_graph_flag = True
                
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)

            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(tag='{}/weight_bp_angle'.format(name),
                                  scalar_value=angles[0],
                                  global_step=step)
                if self.layers[i].use_bias:
                    writer.add_scalar(tag='{}/bias_bp_angle'.format(name),
                                      scalar_value=angles[1],
                                      global_step=step)
            if save_dataframe:
                self.bp_angles.at[step, i] = angles[0].item()

    def save_H_angles(self, writer, step, loss,
                       save_tensorboard=True, save_dataframe=True):
        """Save the angles of the current forward parameter updates
        with the update driven from the Lu loss for those parameters 

        Save the angle in the tensorboard writer (if ``save_tensorboard=True``)
        and in the corresponding dataframe (if ``save_dataframe=True``).

        Args:
            (....): See docstring of method :meth:`save_ndi_angles`
        """
        lu_parameter_updates_W, lu_parameter_updates_b = \
            self.compute_H_update(loss, self.u)

        for i in range(self.depth):
            parameter_update = self.layers[i].get_forward_gradients()
            weights_angle = mutils.compute_angle(lu_parameter_updates_W[i],
                                                 parameter_update[0])
            if self.bias:
                bias_angle = mutils.compute_angle(lu_parameter_updates_b[i],
                                                  parameter_update[1])

            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(tag='{}/weight_lu_angle'.format(name),
                                  scalar_value=weights_angle,
                                  global_step=step)
                if self.bias:
                    writer.add_scalar(tag='{}/bias_lu_angle'.format(name),
                                      scalar_value=bias_angle,
                                      global_step=step)
            if save_dataframe:
                self.lu_angles.at[step,i] = weights_angle.item()
                
        # Compute the total angle between the entire updates across all layers.
        # We only do this for the weights.
        parameter_updates_concat = self.get_vectorized_parameter_updates(
                                                    with_bias=False)
        lu_parameter_updates_concat = mutils.vectorize_tensor_list(\
                                                    lu_parameter_updates_W)
        total_angle = mutils.compute_angle(parameter_updates_concat, 
                                           lu_parameter_updates_concat)
        if save_tensorboard:
            writer.add_scalar(tag='total_alignment/lu_angle',
                              scalar_value=total_angle,
                              global_step=step)
        if save_dataframe:
            self.lu_angles_network.at[step,0] = total_angle.item()#/self.depth

    def save_ratio_ff_fb(self, writer, step, loss,
                               save_tensorboard=True, save_dataframe=True):
        """Save the ratio of the current feedforward and feedback stimulus.

        Save the angle in the tensorboard writer (if ``save_tensorboard=True``)
        and in the corresponding dataframe (if ``save_dataframe=True``).

        Args:
            (....): See docstring of method :meth:`save_ndi_angles`
        """
        ratio = self.compute_ratio_ff_fb(loss, self.u)

        for i in range(self.depth):
            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(tag='{}/ratio_ff_fb'.format(name),
                                  scalar_value=ratio[i],
                                  global_step=step)
            if save_dataframe:
                self.ratio_angle_ff_fb.at[step,i] = ratio[i].item()

        # Compute the total ratio across all layers.
        total_ratio = torch.mean(ratio)
        if save_tensorboard:
            writer.add_scalar(tag='total_ratio_ff_fb',
                              scalar_value=total_ratio,
                              global_step=step)
        if save_dataframe:
            self.ratio_angle_ff_fb_network.at[step,0] = total_ratio.item()


    def save_feedback_batch_logs(self, config, writer, step, init=False,
                                 save_tensorboard=True, save_dataframe=True,
                                 save_statistics=False):
        """Save the logs for the current minibatch on tensorboardX.

        Args:

        Save the angle in the tensorboard writer (if ``save_tensorboard=True``)
        and in the corresponding dataframe (if ``save_dataframe=True``).

        Args:
            (....): See docstring of method :meth:`save_ndi_angles`
            init (bool): Flag indicating that the training is in the
                initialization phase (only training the feedback weights).
            save_statistics: Flag indicating whether the statistics of the
                feedback weights should be saved (e.g. gradient norms).
        """
        if save_statistics:
            for i, layer in enumerate(self.layers):
                name = 'layer_' + str(i+1)
                layer.save_feedback_batch_logs(writer, step, name,
                                     no_gradient=i == 0, pretraining=init)
        if config.save_condition_fb:
            condition_2 = self.compute_condition_two()
            if save_tensorboard:
                if init:
                    writer.add_scalar(tag='feedback_training/condition_2_init',
                                      scalar_value=condition_2,
                                      global_step=step)
                else:
                    writer.add_scalar(tag='feedback_training/condition_2',
                                      scalar_value=condition_2,
                                      global_step=step)
            if save_dataframe:
                if init:
                    self.condition_gn_init.at[step, 0] = condition_2.item()
                else:
                    self.condition_gn.at[step, 0] = condition_2.item()

    def compute_bp_angles(self, loss, i, retain_graph=False):
        """Compute the angles of the current forward parameter updates of layer
        `i` with the backprop update for those parameters.

        Args:
            loss (float): Output loss of the network.
            i (int): Layer index.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns:
            (....): Tuple containing:

            - **weights_angle**: The angle in degrees between the updates for
              the forward weights.
            - **bias_angle**: (Optionally) the angle in degrees for the bias.
        """
        bp_gradients = self.layers[i].compute_bp_update(loss, retain_graph)
        gradients = self.layers[i].get_forward_gradients()

        if mutils.contains_nan(bp_gradients[0].detach()):
            warnings.warn('Backprop update contains NaN (layer {}).'.format(i))
        if mutils.contains_nan(gradients[0].detach()):
            warnings.warn('Weight update contains NaN (layer {}).'.format(i))
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('Norm updates approximately zero (layer {}).'.format(i))
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('Norm updates exactly zero (layer {}).'.format(i))

        weights_angle = mutils.compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].use_bias:
            bias_angle = mutils.compute_angle(bp_gradients[1].detach(),
                                              gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )


    def compute_H_update(self, loss, u):
        r"""Compute the negative weight updates from the :math:`\mathcal{H}`
        loss.

        This computes the gradients w.r.t. the post-nonlinearity activations
        using the full Jacobian of the network:

        .. math::

            \frac{1}{2} \frac{d \lVert Q\mathbf{u}_{ss} \rVert^2_2}{d \
                \bm{\theta}} = -\mathbf{u}_{ss}^{T}Q^{T}Q(J_{ss}Q + \
                \alpha I)^{-1}J_{ss}R_{ss}^{T}

        Args:
            (....): See docstring of method :meth:`compute_bp_update`.
            u (torch.Tensor): The controller signal.

        Returns:
            (....): Tuple containing:

            - **lu_updates_W** (list): The weight updates for each layer
              according to loss :math:`\mathcal{H}`.
            - **lu_updates_b** (list): The bias updates for each layer
              according to loss :math:`\mathcal{H}`. ``None`` if no biases
              exist.
        """
        device = loss.device
        J_ss = self.compute_full_jacobian(noisy_dynamics=True)
        batch_size = J_ss.shape[0]
        Q = self.full_Q
        u_ss = -u
      
        # Create empty variable to save the lu_updates
        lu_updates_b = None
        lu_updates_W = [torch.zeros_like(\
                        self.layers[i].get_forward_gradients()[0],device=device) 
                        for i in range(self.depth)]
        if self.bias:
            lu_updates_b = [torch.zeros_like(\
                            self.layers[i].get_forward_gradients()[1],\
                            device=device) for i in range(self.depth)]

        for b in range(batch_size):

            aux_0 = torch.matmul(torch.matmul(u_ss[b], Q.t()), Q)
            aux_1 = torch.inverse(torch.matmul(J_ss[b],Q) + \
                        self.alpha_di * torch.eye(J_ss[b].shape[0]\
                        ,device=device))
            aux_2 = torch.matmul(torch.matmul(aux_0,aux_1), J_ss[b])

            n = self.layers[0].activations.shape[1]
            aux_3 = aux_2[0 : n]
            aux_4 = self.input[b]
            lu_updates_W[0] += 1./batch_size * mutils.outer(aux_3, aux_4)
            lu_updates_b[0] += 1./batch_size * aux_3

            for i in range(self.depth - 1):
                n_new = n + self.layers[i+1].activations.shape[1]
                aux_3 = aux_2[n:n_new]
                aux_4 = self.layers[i].activations[b]
                lu_updates_W[i+1] += 1./batch_size * mutils.outer(aux_3, aux_4)
                lu_updates_b[i+1] += 1./batch_size * aux_3
                n = n_new
        
        return lu_updates_W, lu_updates_b
    
    def compute_ratio_ff_fb(self, loss, u):
        r"""Compute the ratio of the current feedforward and feedback stimulus.

        It is computed as:

        .. math::
            \frac{||Q_{i}\mathbf{u}(t)||}{||W_{i}\mathbf{r}_{i-1}(t)||}

        Args:
            loss (torch.Tensor): Output loss of the network.
            u (torch.Tensor): Controller signal.

        Returns:
            (list): A list with the ratios for each layer.
        """
        device = u.device
        u_ss = u
        batch_size = u.shape[0]
        W = self.get_forward_parameter_list(with_bias=False)
        
        # empty variable to save the ratios per hidden layer
        ratio_ff_fb = torch.zeros(self.depth,device=device)

        for b in range(batch_size):
            apical = u_ss[b]
            aux_0 = 0
            aux_1 = 0

            Q_0 = self.layers[0].weights_backward
            W_0 = W[0]
            basal = self.input[b]
            aux_0 = torch.norm(torch.matmul(apical, Q_0.t()), p='fro')
            aux_1 = torch.norm(torch.matmul(basal, W_0.t()), p='fro')
            ratio_ff_fb[0] += 1./batch_size * aux_0/aux_1

            for i in range(1, self.depth):

                Q_i = self.layers[i].weights_backward
                W_i = W[i]
                basal = self.layers[i].activations[b]
                aux_0 = torch.norm(torch.matmul(apical, Q_i.t()), p='fro')
                aux_1 = torch.norm(torch.matmul(basal, W_i), p='fro')
                ratio_ff_fb[i] += 1./batch_size * aux_0/aux_1

        return ratio_ff_fb

    def compute_condition_two(self):
        r"""Compute the Gauss-Newton condition on the feedback weights.

        .. math::

            \frac{\|\tilde{J}_2\|_F}{\|\tilde{J}\|_F}

        to keep track whether condition 2 is (approximately) satisfied.
        If the minibatch size is bigger than 1, the mean over the minibatch
        is returned.

        Returns:
            The condition value.
        """
        jacobians = self.compute_full_jacobian(\
                                            noisy_dynamics=self.noisy_dynamics)
        
        Q = self.full_Q
        projected_Q_fro = []

        for b in range(jacobians.shape[0]):
            jac = jacobians[b,:,:]
            projection_matrix = torch.matmul(jac.T,
                torch.matmul(torch.inverse(torch.matmul(jac, jac.T)), jac))

            projected_Q_fro.append(torch.norm(\
                    torch.matmul(projection_matrix, Q), p='fro'))

        projected_Q_fro = torch.stack(projected_Q_fro)
        Q_fro = torch.norm(Q, p='fro')
        condition_two_ratio = projected_Q_fro/Q_fro

        return torch.mean(condition_two_ratio)
