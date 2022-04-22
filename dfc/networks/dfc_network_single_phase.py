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
# @title          :networks/dfc_network_single_phase.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Implementation of a network for Deep Feedback Control that uses a single phase
------------------------------------------------------------------------------

A network that is prepared to be trained with DFC but using a single phase for
training both the forward and the feedback weights.
"""
import torch

from networks.dfc_network import DFCNetwork

class DFCNetworkSinglePhase(DFCNetwork):
    r"""Implementation of a network for Deep Feedback Control with single phase.

    Network that always udpates the feedfoward and feedback weights
    simultaneously in one single phase.

    In this single-phase DFC setting, the following options exist for defining
    the target activations. For forward weight learning, the target outputs are
    either feedforward activations nudged towards lower loss (default) or set as
    the actual supervised targets (if the option `strong_feedback` is active),
    just like in two-phase DFC. For feedback weight learning, the target
    outputs are either nudged or set to the supervised targets (if
    `strong_feedback` is active). However, in the pre-training stage, if the
    option `pretrain_without_controller` is active, the targets are set to the
    forward activations.

    Args:
        (....): See docstring of class :class:`dfc_network.DFCNetwork`.
        pretrain_without_controller (bool): Whether pretraining should be done
            without the controller being on.
    """
    def __init__(self, *args, pretrain_without_controller=False, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine constants to filter out the control signal and dynamics.
        self._alpha_u = self.dt_di / self.tau_f
        self._alpha_r = self.dt_di / self.tau_f

        # We always low-pass filter the noise, and therefore we don't need to
        # simulate the apical compartment dynamics.
        self._inst_apical_dynamics = True # apical_time_constant is unneeded
        self._low_pass_filter_noise = True 
        self._pretrain_without_controller = pretrain_without_controller

    @property
    def pretrain_without_controller(self):
        """ Getter for read-only attribute
        :attr:`pretrain_without_controller`"""
        return self._pretrain_without_controller

    @property
    def alpha_r(self):
        """Getter for read-only attribute :attr:`alpha_r`."""
        return self._alpha_r

    @property
    def alpha_u(self):
        """Getter for read-only attribute :attr:`alpha_u`."""
        return self._alpha_u

    def backward(self, loss, targets=None, verbose=True):
        """Run the feedback phase of the network.

        Here, the network is pushed to the output target by the controller and
        used to compute update of the forward and backward weights.

        This function simply constitutes a wrapper around the base `backward`
        function, where forward updates are computed, and just adds the
        feedback weight update.
        
        Args:
            loss (torch.Tensor): Mean output loss for current mini-batch.
            targets (torch.Tensor): The dataset targets. This will usually be
                ignored, as the targets will be taken to be the activations
                nudged towards lower loss, unless we use strong feedback.
            verbose (bool): Whether to display warnings.
        """
        ### Compute the feedforward gradients.
        u_time, v_fb_time = super().backward(loss, targets, return_for_fb=True,
                                             verbose=verbose)

        if not self.freeze_fb_weights and not self.use_jacobian_as_fb:
            ### Compute the feedback gradients.
            self.compute_feedback_gradients(loss, targets, u_time, v_fb_time)
        
    def dynamical_inversion(self, *args, **kwargs):
        """Compute the dynamical (simulated) inversion of the targets.

        Applies the same function as in the base DFC class, but adds a low-pass
        filter to the target activations.

        Args:
            output_target (torch.Tensor): The output targets.
        """
        u_ss, v_ss, r_ss, target_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r) = \
                    super().dynamical_inversion(*args, **kwargs)
        device = u_ss.device

        # Compute lowpass filter of r (average out the injected noise).
        # Note that this function is only called within `backward`, i.e. when
        # the forward weights are being trained, and so we can call `alpha_r`
        # which makes use of the forward training `dt_di` value.
        if self.noisy_dynamics:
            r_lp = [torch.zeros_like(val, device=device) for val in r]
            for l in range(self.depth):
                r_lp[l][0] = r[l][0].clone()
                for t in range(1, int(self.tmax_di)):
                    r_lp[l][t] = self.alpha_r * r[l][t] + \
                                    (1 - self.alpha_r) * r_lp[l][t - 1]
        else:
            r_lp = r

        # Get the steady states.
        r_ss = [val[-1] for val in r_lp]
        r_out_ss = r_ss[-1]

        return u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r)

    def compute_feedback_gradients(self, loss, targets, u_time=None,
                                   v_fb_time=None, init=False):
        r"""Compute the gradients of the feedback weights for each layer.

        This function is called in two different situations:

        1. During pre-training of the feedback weights, there has not yet been a
        simulation, so this function calls a simulation (with special values for
        :math:`\alpha` and :math:`k` to ensure stability during pre-training) and
        uses the results of the simulation to update the feedback weights.
        In this case, the inputs :math:`\mathbf{v}^\text{fb}` and 
        :math:`\mathbf{u}^\text{hp}` will be ``None``.

        2. During the simultaneous training of feedforward and feedback weights,
        the backward method already simulates the dynamics, and the results are
        passed through :math:`\mathbf{v}^\text{fb}` and
        :math:`\mathbf{u}^\text{hp}`. In this case, we directly use these
        simulation results to compute the updates without running a new
        simulation.

        The feedback weight updates are computed according to the following rule:
        
        .. math::

            \Delta Q = -(1+\frac{\tau_v}{\tau_{\epsilon}})^{L-i}\
                \frac{1}{K\sigma^2} \sum_k \mathbf{v}^\text{fb}_i[k] \
                \mathbf{u}^{\text{hp}T}[k]

        Args:
            (....): See docstring of method :meth:`backward`.
            u_time (torch.Tensor): A torch.Tensor of dimension 
                :math:`t_{max}\times B \times n_L` containing the high-pass
                filtered controller inputs. If None (by default), a new
                simulation will be run to calculate v_fb_time and u_time.
            v_fb_time (torch.Tensor): A list with at index ``i`` a torch.Tensor
                of dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical (feedback) compartment of layer
                `i`. If ``None`` (by default), a new simulation will be run to
                calculate :math:`\mathbf{v}^\text{fb}` and
                :math:`\mathbf{u}^\text{hp}`.
            init (bool): Whether this is a pre-training stage. If ``True``,
                dynamics values specific for the feedback path will be used.
                Else, the same as the forward pass will be used.
        """
        # Suffix to select the right simulation parameter depending on whether
        # we are in the common ff and fb training phase or not.
        sf = ''
        sigma_output = self.sigma

        if u_time is None:
            # Only backward weights are trained (pre-training or extra fb epoch)
            sf = '_fb'
            sigma_output = self.sigma_output_fb

            assert v_fb_time is None

            # We define the target in the same way as for learning ff weights,
            # except if we are pre-training and the option
            # `pretrain_without_controller` is on, in which case we use the
            # same targets as in the two-phase setting.
            output_target = self.compute_output_target(loss, targets)
            if init and self.pretrain_without_controller:
                output_target = self.layers[-1].activations.data

            # Compute the controller signal.
            if init:
                # When only the feedback weights are being trained, we can set
                # all the simulation hyperparameters to their backward version.
                _, u_time, (v_fb_time, _, _), _ =  \
                    self.controller(output_target=output_target,
                                alpha=self.alpha_di_fb,
                                dt=self.dt_di_fb,
                                tmax=self.tmax_di_fb,
                                k_p=self.k_p_fb,
                                noisy_dynamics=True,
                                inst_transmission=self.inst_transmission_fb,
                                time_constant_ratio=self.time_constant_ratio_fb,
                                proactive_controller=self.proactive_controller,
                                sigma=self.sigma_fb,
                                sigma_output=self.sigma_output_fb)
            else:
                _, u_time, (v_fb_time, _, _), _ =  \
                    self.controller(output_target=output_target,
                                alpha=self.alpha_di,
                                dt=self.dt_di,
                                tmax=self.tmax_di,
                                k_p=self.k_p,
                                noisy_dynamics=True,
                                inst_transmission=self.inst_transmission,
                                time_constant_ratio=self.time_constant_ratio,
                                proactive_controller=self.proactive_controller,
                                sigma=self.sigma,
                                sigma_output=self.sigma_output)

        ### Get high-pass filtered control.
        # Compute lowpass filter of u using exponential smoothing.
        u_time = u_time[1:, :, :]
        u_aux = torch.zeros_like(u_time)
        u_aux[0] = u_time[0]
        for t in range(1, len(u_time)):
            u_aux[t] = self.alpha_u * u_time[t] + (1-self.alpha_u) * u_aux[t-1]

        # Subtract the low-pass signal to obtain high-pass signal.
        # This gets rid of the average target of the nudging phase, and keeps
        # only the noise needed to learn the feedback weights.
        u_filtered = u_time - u_aux

        # Extract some important parameters that need to be used later.
        time_constant_ratio = getattr(self, 'time_constant_ratio', sf)
        sigma = getattr(self, 'sigma' + sf)
        sigma_output = getattr(self, 'sigma_output' + sf)

        # Compute gradient for each layer.
        for i, layer in enumerate(self.layers):
            v_fb_i = v_fb_time[i][:-1, :, :]

            # compute a layerwise scaling for the feedback weights
            scaling = 1.
            if self.scaling_fb_updates:
                scaling = (1 + time_constant_ratio / self.tau_noise) \
                         ** (len(self.layers) - i - 1)

            # get the amount of noise used.
            sigma_i = sigma
            if i == len(self.layers) - 1:
                sigma_i = sigma_output

            layer.compute_feedback_gradients_continuous(v_fb_i, u_filtered,
                                                        sigma=sigma_i,
                                                        scaling=scaling)
