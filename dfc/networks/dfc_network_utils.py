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
# @title          :networks/dfc_network_utils.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
Script with helper functions for Deep Feedback Control computations
-------------------------------------------------------------------

This module contains several helper functions for training with DFC.
"""
from hypnettorch.utils import torch_ckpts as ckpts
import numpy as np
import os
import torch
import torch.nn as nn

from utils.optimizer_utils import extract_parameters
from utils import sim_utils

def train_feedback_parameters(config, logger, writer, device, dloader, net,
                              optimizers, shared, loss_function,
                              pretraining=False):
    """Train the feedback weights.

    This function is called either to perform further training of feedback
    weights after each epoch of forward parameter training, or as a pre-training
    to initialize the network in a 'pseudo-inverse' condition.

    Args:
        config (Namespace): The command-line arguments.
        logger: The logger.
        writer: The writer.
        dloader: The dataset.
        net: The neural network.
        optimizers: The optimizers.
        shared: The Namespace containing important training information.
        loss_function: The loss function.
        pretraining (boolean): Whether the call is for pretraining or not.
    """
    if pretraining:
        epochs = config.init_fb_epochs
        optimizer = optimizers['feedback_init']
        prefix = 'Pre-'
        if epochs == 0:
            logger.info('Feedback weights are not being trained.')
    else:
        epochs = config.extra_fb_epochs
        optimizer = optimizers['feedback']
        prefix = 'Extra-'

    fb_training = epochs != 0 and not config.freeze_fb_weights

    if fb_training:
        logger.info('Feedback weight %straining...' % prefix.lower())
        for e in range(epochs):
            logger.info('     %straining epoch %i/%i...' %(prefix, e+1,
                                                           epochs))

            # Only compute condition if required and pre-training, since in
            # normal training it is computed in the forward training function.
            train_epoch_feedback(config, logger, writer, dloader, optimizer,
                                 net, shared, loss_function, epoch=e,
                                 pretraining=pretraining,
                                 compute_gn_condition=\
                                    config.save_condition_fb and pretraining)

            if net.contains_nans():
                logger.info('Network contains NaN: terminating %straining.'\
                            % prefix.lower())
                break

        logger.info('Feedback weight %straining... Done' % prefix.lower())

        # Save the pre-trained network.
        if pretraining and config.save_checkpoints:
            ckpts.save_checkpoint({'state_dict': net.state_dict,
                                   'net_state': 'pretrained'},
                   os.path.join(config.out_dir, 'ckpts/pretraining'), None)

def train_epoch_feedback(config, logger, writer, dloader, optimizer, net,
                         shared, loss_function, epoch=None, pretraining=False,
                         compute_gn_condition=False):
    """Train the feedback parameters for one epoch.

    For each mini-batch in the training set, this function:

    * computes the forward pass
    * sets the feedback gradients to zero and computes the gradients
    * clips the feedback gradients if necessary
    * updates the feedback weights

    Args:
        config: The command-line config.
        logger: The logger.
        writer: The writer.
        dloader: The data loader.
        optimizer: The feedback optimizer.
        net: The network.
        shared: The Namespace containing important training information.
        loss_function: The loss function.
        epoch (int): The current epoch.
        pretraining (boolean): Whether the call is for pretraining or not.
        compute_gn_condition (boolean): Whether to compute the gn condition
            during this epoch or not.
    """
    # Iterate over the dataset.
    for i, (inputs, targets) in enumerate(dloader.train):
        predictions = net.forward(inputs)

        # We need to compute the loss for the case of the single-phase, where
        # the targets will be set as the nudged loss, like in forward learning.
        loss = loss_function(predictions, targets)

        ### Compute gradients and update weights.
        optimizer.zero_grad() # check. should this be after predictions?
        net.compute_feedback_gradients(loss, targets, init=pretraining)
        if config.clip_grad_norm != -1:
            for param in extract_parameters(net, config, 'DFC',
                                            params_type='feedback'):
                nn.utils.clip_grad_norm_(param, max_norm=config.clip_grad_norm)
            if np.isnan(net.get_max_grad(params_type='feedback').item()):
                raise ValueError('NaN encountered during feedback training.')
            assert net.get_max_grad(params_type='feedback') <= \
                config.clip_grad_norm
        optimizer.step()
        
        if pretraining:
            shared.train_var.batch_idx_fb_init += 1
        else:
            shared.train_var.batch_idx_fb += 1

        if config.test and i == 1:
            break

    # If required, compute the gn condition on the feedback weights.
    if pretraining:
        sim_utils.log_stats_to_writer(config, writer, epoch+1, net, init=True)

    # Compute the condition at the very end of the epoch.
    if pretraining and config.save_condition_fb:
        condition_gn = net.compute_condition_two()
        shared.train_var.gn_condition_init.append(condition_gn)
        logger.info('     Condition 2: %.3f.' % condition_gn)

def loss_function_H(config, net, shared):
    r"""Compute surrogate :math:`\mathcal{H}` loss on the last batch.

    This loss corresponds to the norm of the total amount of help, computed
    as :math:`||Q\mathbf{u}||^2`, normalized by the batch size and the
    number of neurons.

    Args:
        config: The config.
        net: The network.
        shared: The shared subspace.

    Returns:
        (float): The normalized :math:`\mathcal{H}` loss.
    """
    J = net.compute_full_jacobian(noisy_dynamics=True)
    u_ss = net.u
    batchsize = u_ss.shape[0]

    loss_lu = 0
    for b in range(batchsize):
        if config.use_jacobian_as_fb:
            feedback_weights = J[b, :, :].t()
        else:
            feedback_weights = net.full_Q
        loss_lu += torch.norm(torch.matmul(feedback_weights,
                                           u_ss[b,:].unsqueeze(1)))**2

    # Divide by total number of neurons in the network (except input).
    num_neurons = np.sum([layer.weights.data.shape[0] for layer in net.layers])

    return loss_lu / (batchsize * num_neurons)

def save_angles(config, writer, step, net, loss):
    """Save logs and plots for the current mini-batch on tensorboard.

    Args:
        config (Namespace): The config.
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        loss (torch.Tensor): loss of the current minibatch.
    """
    if config.save_bp_angle:
        retain_graph = config.save_H_angle
        net.save_bp_angles(writer, step, loss, retain_graph=retain_graph,
                           save_tensorboard=not config.no_plots,
                           save_dataframe=config.save_df)

    if config.save_H_angle:
        net.save_H_angles(writer, step, loss,
                           save_tensorboard=not config.no_plots,
                           save_dataframe=config.save_df)

    if config.save_ratio_ff_fb:
        net.save_ratio_ff_fb(writer, step, loss,
                             save_tensorboard=not config.no_plots,
                             save_dataframe=config.save_df)
    
    if config.save_ndi_angle:
        net.save_ndi_angles(writer, step, save_tensorboard=not config.no_plots,
                            save_dataframe=config.save_df)