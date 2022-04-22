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
# @title          :utils/optimizer_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for building optimizers
----------------------------------------

A collection of functions for building a custom set of optimizers.
"""
import numpy as np
import torch

def get_optimizers(config, net, network_type='BP', logger=None):
    """Return the optimizers based on command line arguments.

    Returns optimizers for forward weights and when necessary of feedback and
    feedback weight initialization as well.

    Args:
        config: The command-line arguments.
        net: The network.
        network_type (str): The type of network.
        logger: The logger. If `None` nothing will be logged.

    Returns:
        (dict): A dictionary containing forward, feedback and feedback init
            optimizers, if required.
    """
    if logger is not None:

        if hasattr(config, 'only_train_last_layer') and \
                                                config.only_train_last_layer:
            logger.info('Shallow training.')
        if hasattr(config, 'only_train_first_layer') and \
                                                config.only_train_first_layer:
            logger.info('Only training first layer.')

    ### Construct forward optimizer.
    forward_params = extract_parameters(net, config, network_type,
                                        return_nones=True)
    forward_optimizer = OptimizerList(forward_params,
                          lr=config.lr, 
                          optimizer_type=config.optimizer,
                          no_bias=config.no_bias,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay,
                          adam_beta1=config.adam_beta1,
                          adam_beta2=config.adam_beta2,
                          adam_epsilon=config.adam_epsilon)
    if logger is not None and not config.freeze_fw_weights:
        forward_optimizer.log_info(logger, opt_loc='forward')
    optimizers = {'forward': forward_optimizer}

    ### Construct backward optimizers.
    if 'DFC' in network_type:
        feedback_params = extract_parameters(net, config, network_type,
                                             params_type='feedback',
                                             return_nones=True)
        
        feedback_optimizer = OptimizerList(feedback_params,
                          lr=config.lr_fb, 
                          optimizer_type=config.optimizer_fb,
                          no_bias=True,
                          momentum=config.momentum_fb,
                          weight_decay=config.weight_decay_fb,
                          adam_beta1=config.adam_beta1_fb,
                          adam_beta2=config.adam_beta2_fb,
                          adam_epsilon=config.adam_epsilon_fb)
        if logger is not None and not config.freeze_fb_weights:
            feedback_optimizer.log_info(logger, opt_loc='feedback')
        optimizers['feedback'] = feedback_optimizer

        feedback_optimizer_init = OptimizerList(feedback_params,
                          lr=config.lr_fb_init, 
                          optimizer_type=config.optimizer_fb,
                          no_bias=True,
                          momentum=config.momentum_fb,
                          weight_decay=config.weight_decay_fb,
                          adam_beta1=config.adam_beta1_fb,
                          adam_beta2=config.adam_beta2_fb,
                          adam_epsilon=config.adam_epsilon_fb)
        if logger is not None and not config.freeze_fb_weights:
            feedback_optimizer_init.log_info(logger, opt_loc='feedback init')
        optimizers['feedback_init'] = feedback_optimizer_init

    return optimizers

def extract_parameters(net, config, network_type, params_type='forward',
                       return_nones=False):
    """Extract list of parameters to be optimized.

    This function can utilize the native functions of the networks that directly
    provide the list of parameters, but here we additionally look at the command
    line arguments and see whether some options freeze certain of those
    parameters.

    By default, a list of the parameters to be learned is returned. However, if
    the option `return_nones` is activated, the list of parameters might have
    `None` for those layer parameters that exist but shouldn't be learned. This
    is useful, for example, when generating the optimizer in case there is a
    per-layer learning rate.

    Args:
        net: The network.
        config: The command-line arguments.
        network_type (str): The type of network.
        params_type (str): The type of parameters, for DFC networks. Can be
            `forward` or `feedback`.
        return_nones (boolean): 

    Returns:
        (list): The parameters to be optimized.
    """
    if network_type == 'BP':
        # For backprop, plasticity might be limited to certain layers.
        params = net.params
        if return_nones:
            for i, p in enumerate(params):
                if (i != 0 and config.only_train_first_layer) or \
                        config.freeze_fw_weights or \
                        (i != len(params)-1 and config.only_train_last_layer):
                    params[i] = None
        else:
            if config.freeze_fw_weights:
                params = []
            elif config.only_train_last_layer:
                params = params[-1]
            elif config.only_train_first_layer:
                params = params[0]
            
    elif network_type == 'DFA':
        # For DFA, the network only returns the forward parameters.
        params = net.params
        if return_nones:
            for i, p in enumerate(params):
                if config.freeze_fw_weights:
                    params[i] = None
        else:
            if config.freeze_fw_weights:
                params = []

    elif 'DFC' in network_type:
        # For DFC, we need to specify which parameters we want to train.
        if params_type == 'forward':
            params = net.forward_params
            if return_nones:
                for i, p in enumerate(params):
                    if config.freeze_fw_weights:
                        params[i] = None
            else:
                if config.freeze_fw_weights:
                    params = []
        elif params_type == 'feedback':
            params = net.feedback_params
            if return_nones:
                if config.freeze_fb_weights:
                    for i, p in enumerate(params):
                        params[i] = None
                elif config.freeze_fb_weights_output:
                    params[-1] = None
            else:
                if config.freeze_fb_weights:
                    params = []
                elif config.freeze_fb_weights_output:
                    params[-1] = []

    return params

class OptimizerList(object):
    """An optimizer instance that handles layer-specific specifications.

    This class stacks a separate optimizer for each layer in a list. If
    no separate learning rates per layer are required, a single optimizer is
    stored in the optimizer list.

    Args:
        params_list (list): The parameters to be optimized.
        lr (float): The learning rate.
        network_type (str): The type of network.
        optimizer_type (str): The optimizer type.
        no_bias (boolean): Whether no bias terms are learned.
        momentum (boolean): The momentum value.
        forward_wd (boolean): The forward weight decay value.
        adam_beta1 (float): beta1 value for Adam.
        adam_beta2 (float): beta2 value for Adam.
        adam_epsilon (float): epsilon value for Adam.
    """
    def __init__(self, params_list, lr=1e-3, optimizer_type='SGD',
                 network_type='BP', no_bias=False, momentum=0,
                 weight_decay=0, adam_beta1=0.99, adam_beta2=0.99,
                 adam_epsilon=1e-8):

        if not isinstance(params_list, list):
            params_list = [params_list]
        if isinstance(adam_epsilon, float):
            adam_epsilon = [adam_epsilon]*len(params_list)
        elif isinstance(adam_epsilon, list) and len(adam_epsilon) == 1:
            adam_epsilon = [adam_epsilon[0]]*len(params_list)
        if isinstance(lr, float):
            lr = [lr]*len(params_list)
        elif isinstance(lr, list) and len(lr) == 1:
            lr = [lr[0]]*len(params_list)

        optimizer_list = []
        for i, params in enumerate(params_list):
            if not params is None:
                optimizer_list.append(build_optimizer(params, lr=lr[i],
                                            optimizer_type=optimizer_type,
                                            weight_decay=weight_decay,
                                            momentum=momentum,
                                            adam_beta1=adam_beta1,
                                            adam_beta2=adam_beta2,
                                            adam_epsilon=adam_epsilon[i]))

        self._lr = lr
        self._optimizer_type = optimizer_type
        self._optimizer_list = optimizer_list
        self.parameters = params_list
        self.name = optimizer_type

    def zero_grad(self):
        """Set all the gradients to zero."""
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()

    def step(self, i=None):
        """Perform a step on the optimizer in all or a single specific layer.

        Args:
            i (int): The layer where to perform the step. If ``None``, then the
                step is made on all optimizers.
        """
        if i is None:
            for optimizer in self._optimizer_list:
                optimizer.step()
        else:
            self._optimizer_list[i].step()

    def __getitem__(self, i):
        """Overwrite get item function.

        Args:
            i (int): The index of desired element of the optimizer list.

        Return:
            The corresponding optimizer.
        """
        return self._optimizer_list[i]

    def log_info(self, logger, opt_loc=''):
        """Display information about optimizer.

        Args:
            logger: The logger.
            opt_loc (str): The type of optimizer (forward, feedback...).
        """
        if len(np.unique(self._lr)) == 1:
            logger.info('Using %s %s optimizer with lr = %.5f.' % \
                        (self.name, opt_loc, self._lr[0]))
        else:
            logger.info('Using %s %s optimizer with:' % (self.name, opt_loc))
            for forward_opt in self._optimizer_list:
                assert len(forward_opt.param_groups) == 1
                lr = forward_opt.param_groups[0]['lr']
                shapes = str([list(pm.shape) for pm in \
                            forward_opt.param_groups[0]['params']])
                logger.info('     lr = %.3f for params with shape %s.' % \
                            (lr, shapes[1:-1]))


def build_optimizer(params, optimizer_type='SGD', lr=1e-3,
                     weight_decay=0, momentum=0, adam_beta1=0.99,
                     adam_beta2=0.99, adam_epsilon=1e-8):
    """Build optimizer given a certain set of parameters to be optimized.

    This function can be used for building forward and backward optimizers.

    Args:
        params: The parameters to be optimized.
        optimizer_type (str): The name of the optimizer.
        lr (float): The learning rate.
        weight_decay (float): The weight decay.
        momentum (float): The momentum for SGD and RMSprop optimizers.
        adam_beta1 (float): beta1 value for Adam.
        adam_beta2 (float): beta2 value for Adam.
        adam_epsilon (float): epsilon value for Adam.

    Returns:
        The optimizer.
    """
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum,
                                        alpha=0.95,
                                        eps=0.03,
                                        weight_decay=weight_decay,
                                        centered=True)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr,
                                     betas=(adam_beta1, adam_beta2),
                                     eps=adam_epsilon,
                                     weight_decay=weight_decay)

    return optimizer