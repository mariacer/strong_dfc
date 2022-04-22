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
# @title          :networks/net_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for generating different networks
--------------------------------------------------

A collection of helper functions for generating networks to keep other scripts
clean.
"""
from networks import bp_network, dfa_network, dfc_network,\
                     dfc_network_single_phase

def generate_network(config, dataset, device, network_type='BP',
                     classification=True, logger=None):
    """Create the network based on the provided command line arguments.

    config:
        config: Command-line arguments.
        dataset: The dataset being used.
        device: The cuda device.
        network_type (str): The type of network.
        classification (boolean): Whether the task is a classification task.
        logger: The logger. If `None` nothing will be logged.

    Returns:
        The network.
    """
    # Prepare the necessary keywords.
    kwconfig_bp = {
        'n_in': dataset.in_size,
        'n_hidden': config.size_hidden,
        'n_out': dataset.out_size,
        'activation': config.hidden_activation,
        'bias': not config.no_bias,
        'initialization': config.initialization
    }

    # Generate the network.
    if network_type == 'BP':
        net = bp_network.BPNetwork(**kwconfig_bp)

    elif network_type == 'DFA':
        net = dfa_network.DFANetwork(**kwconfig_bp) 

    elif network_type == 'DFC' or network_type == 'DFC_single_phase':
        forward_requires_grad = config.save_bp_angle or config.compare_with_ndi
        kwconfig_dfc_base = {
                'sigma' : config.sigma, 
                'sigma_fb' : config.sigma_fb,
                'sigma_output' : config.sigma_output,
                'sigma_output_fb' : config.sigma_output_fb,
                'sigma_init': config.sigma_init,
                'epsilon_di': config.epsilon_di,
                'initialization_fb': config.initialization_fb,
                'alpha_di': config.alpha_di,
                'alpha_di_fb': config.alpha_di_fb,
                'dt_di': config.dt_di,
                'dt_di_fb': config.dt_di_fb,
                'tmax_di': config.tmax_di,
                'tmax_di_fb': config.tmax_di_fb,
                'k_p': config.k_p,
                'k_p_fb': config.k_p_fb,
                'inst_transmission': config.inst_transmission,
                'inst_transmission_fb': config.inst_transmission_fb,
                'time_constant_ratio': config.time_constant_ratio,
                'time_constant_ratio_fb': config.time_constant_ratio_fb,
                'proactive_controller': config.proactive_controller,
                'noisy_dynamics': config.noisy_dynamics,
                'inst_system_dynamics': config.inst_system_dynamics,
                'inst_transmission_fb': config.inst_transmission_fb,
                'target_stepsize': config.target_stepsize,
                'include_non_converged_samples': \
                    not config.include_only_converged_samples,
                'compare_with_ndi': config.compare_with_ndi,
                'save_ndi_updates': config.save_ndi_angle,
                'save_df': config.save_df,
                'low_pass_filter_u': config.low_pass_filter_u,
                'low_pass_filter_noise': config.low_pass_filter_noise,
                'tau_f': config.tau_f,
                'tau_noise': config.tau_noise,
                'use_jacobian_as_fb': config.use_jacobian_as_fb,
                'freeze_fb_weights': config.freeze_fb_weights,
                'scaling_fb_updates': config.scaling_fb_updates,
                'compute_jacobian_at': config.compute_jacobian_at,
                'sigma_fb' : config.sigma_fb,
                'sigma_output_fb' : config.sigma_output_fb,
                'alpha_di_fb': config.alpha_di_fb,
                'dt_di_fb': config.dt_di_fb,
                'tmax_di_fb': config.tmax_di_fb,
                'k_p_fb': config.k_p_fb,
                'inst_transmission_fb': config.inst_transmission_fb,
                'time_constant_ratio_fb': config.time_constant_ratio_fb,
                'learning_rule': config.learning_rule,
                'strong_feedback': config.strong_feedback}

        if network_type == 'DFC':
            cont_updates = not (config.ssa or config.ss)
            forward_requires_grad = forward_requires_grad or config.ssa

            if logger is not None:
                if config.ss:
                    logger.info('Steady-state updates computed dynamically.')
                elif config.ssa:
                    logger.info('Steady-state updates computed with analytical '
                                'solution.') # old non-dynamical inversion
                else:
                    logger.info('Continuous updates computed dynamically.')
                    if config.compare_with_ndi:
                        logger.info('Also computing analytical solution for '
                                    'comparison (this causes a computational '
                                'overhead).')
            kwconfig_dfc = {'ndi': config.ssa,
                    'cont_updates': cont_updates,
                    'apical_time_constant': config.apical_time_constant,
                    'apical_time_constant_fb': config.apical_time_constant_fb,
                    'forward_requires_grad': forward_requires_grad}

            net = dfc_network.DFCNetwork(**kwconfig_bp, **kwconfig_dfc,
                                         **kwconfig_dfc_base) 

        elif network_type == 'DFC_single_phase':
            logger.info('Single-phase updates computed dynamically.')
            kwconfig_dfc = {'ndi': False,
                            'cont_updates': True,
                            'forward_requires_grad': forward_requires_grad,
                            'pretrain_without_controller': \
                                       config.pretrain_without_controller}

            net = dfc_network_single_phase.DFCNetworkSinglePhase(
                                                    **kwconfig_bp,
                                                    **kwconfig_dfc,
                                                    **kwconfig_dfc_base)  
    else: 
        raise ValueError('The provided network type {} is not supported'.\
                format(network_type))

    # Print summary information about the network.
    if logger is not None:
        log_net_details(logger, net, network_type)

    net.to(device)

    return net

def log_net_details(logger, net, network_type=None):
    """Log the architecture of the network.

    Args:
        logger: The logger.
        net: The network.
        network_type: The type of network.
    """
    shapes = []
    for param in net.forward_params:
        if len(param) == 2:
            bias = 'with'
        elif len(param) == 1:
            bias = 'without'
        shape = [list(pp.shape) for pp in param]
        shapes.extend(shape)
    if network_type is not None:
        logger.info('Created %s network %s bias.' % (network_type, bias))
    logger.info('Network architecture: %s' % str(shapes))
