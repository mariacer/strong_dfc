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
# @title          :utils/train_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for training and testing networks
--------------------------------------------------

A collection of functions for training and testing networks.
"""
from hypnettorch.utils import torch_ckpts as ckpts
import numpy as np
import os
import time
import torch
import torch.nn as nn
import warnings

from networks import dfc_network_utils as dfc
from utils import math_utils as mutils
from utils import plt_utils
from utils import sim_utils
from utils.optimizer_utils import extract_parameters

def train(config, logger, device, writer, dloader, net, optimizers, shared,
          network_type, loss_fn):
    """Train the network.

    Args:
        config: The command-line arguments.
        logger: The logger.
        device: The cuda device.
        writer: The tensorboard writer.
        dloader: The dataset.
        net: The network.
        optimizers: The optimizers.
        shared: Shared object with task information.
        network_type (str): The type of network.
        loss_fn: The loss function.

    Return:
        (dict): The shared object containing summary information.
    """
    if config.test:
        logger.info('Option "test" is active. This is a dummy run!')

    logger.info('Training network ...')
    net.train()
    net.zero_grad()

    # If the error needs to be computed as the gradient of the loss within 
    # the network, provide the name of the loss function being used.
    if hasattr(config, 'error_as_loss_grad') and config.error_as_loss_grad:
        if shared.classification:
            loss_function_name = 'cross_entropy'
        else:
            loss_function_name = 'mse'
        net.loss_function_name = loss_function_name

    for e in range(config.epochs):
        logger.info('Training epoch %i/%i...' % (e+1, config.epochs))
        epoch_initial_time = time.time()

        ### Train.
        # Feedback training for two-phase DFC.
        if network_type == 'DFC' and not config.freeze_fb_weights:
            dfc.train_epoch_feedback(config, logger, writer, dloader,
                                     optimizers['feedback'], net, shared,
                                     loss_fn, epoch=e)

        # Forward training.
        epoch_losses, epoch_accs = train_epoch_forward(config, logger,
                                               device, writer, shared, dloader,
                                               net, optimizers, loss_fn,
                                               network_type, epoch=e)

        # If required train feedback weights for extra epochs.
        if 'DFC' in network_type and not config.freeze_fb_weights:
            dfc.train_feedback_parameters(config, logger, writer, device,
                                          dloader, net, optimizers, shared,
                                          loss_fn)

        ### Test.
        epoch_test_loss, epoch_test_accu = test(config, logger, device, writer,
                                                shared, dloader, net, loss_fn,
                                                network_type)

        ### Validate.
        if not config.no_val_set:
            epoch_val_loss, epoch_val_accu = test(config, logger, device,
                                                  writer, shared, dloader, net,
                                                  loss_fn, network_type,
                                                  data_split='validation')

        # Keep track of performance results.
        epoch_time = np.round(time.time() - epoch_initial_time)
        shared.train_var.epochs_time.append(epoch_time)

        # Log some information.
        logger.info('     Test loss: %.4f' % epoch_test_loss)
        if shared.classification:
            logger.info('     Test accu: %.2f%%' % (epoch_test_accu*100))
        logger.info('     Time %i s' % epoch_time)

        # Write summary information.
        shared = sim_utils.update_summary_info(config, shared, network_type)

        # Add results to the writer.
        sim_utils.add_summary_to_writer(config, shared, writer, e+1)
        sim_utils.log_stats_to_writer(config, writer, e+1, net)

        # Same the performance summary.
        sim_utils.save_summary_dict(config, shared)
        if config.epoch_summary_interval != -1 and \
                                        e % config.epoch_summary_interval == 0:
            # Every few epochs, save separate summary file.
            sim_utils.save_summary_dict(config, shared, epoch=e)

        if net.contains_nans():
            logger.info('Network contains NaNs, terminating training.')
            shared.summary['finished'] = -1
            break

        # Save the training network.
        if e % config.checkpoint_interval == 0 and config.save_checkpoints:
            store_dict = {'state_dict': net.state_dict,
                          'net_state': 'epoch_%i' % e,
                          'train_loss': shared.train_var.epochs_train_loss[-1],
                          'test_loss': shared.train_var.epochs_test_loss[-1]}
            if shared.classification:
                store_dict['train_acc'] = shared.train_var.epochs_train_accu[-1]
                store_dict['test_acc'] = shared.train_var.epochs_test_accu[-1]
            ckpts.save_checkpoint(store_dict,
                    os.path.join(config.out_dir, 'ckpts/training'), None)

        # Kill the run if results are below desired threshold.
        if shared.classification and e == 3 and epoch_accs[-1] < config.min_acc:
            logger.info('Simulation killed: low accuracy at epoch %i.'%(e+1))
            shared.summary['finished'] = -1
            break

    # Save the final network.
    if config.save_checkpoints:
        store_dict = {'state_dict': net.state_dict,
                      'net_state': 'trained',
                      'train_loss': shared.train_var.epochs_train_loss[-1],
                      'test_loss': shared.train_var.epochs_test_loss[-1]}
        if shared.classification:
            store_dict['train_acc'] = shared.train_var.epochs_train_accu[-1]
            store_dict['test_acc'] = shared.train_var.epochs_test_accu[-1]
        ckpts.save_checkpoint(store_dict,
                              os.path.join(config.out_dir, 'ckpts/final'), None)

    # Finish up the training.
    if shared.summary['finished'] == 0:
        # Only overwrite if the training hasn't been stopped due to NaNs.
        shared.summary['finished'] = 1
    logger.info('Training network ... Done.')

    return shared

def train_epoch_forward(config, logger, device, writer, shared, dloader, net,
                        optimizers, loss_fn, network_type, epoch=None):
    """Train forward weights for one epoch.

    For backpropagation, remember that forward and feedback parameters are one
    and the same, so this function is equivalent to normal training.

    Args:
        config (Namespace): The command-line arguments.
        logger: The logger.
        device: The PyTorch device to be used.
        writer (SummaryWriter): TensorboardX summary writer to save logs.
        shared: Shared object with task information.
        dloader: The dataset.
        net: The neural network.
        optimizers (dict): The optimizers.
        loss_fn: The loss function to use.
        network_type (str): The type of network.
        epoch: The current epoch.

    Returns:
        (....): Tuple containing:

        - **epoch_losses**: The list of losses in all batches of the epoch.
        - **epoch_accs**: The list of accuracies in all batches of the epoch.
            ``None`` for non classification tasks.
    """
    epoch_losses = []
    epoch_accs = [] if shared.classification else None
    single_phase = network_type == 'DFC_single_phase'

    # Do we need to compute the gradients in this function?
    compute_gradient = not config.freeze_fw_weights
    if single_phase:
        compute_gradient = compute_gradient or not config.freeze_fb_weights

    if 'DFC' in network_type:
        if config.save_lu_loss:
            epoch_loss_lu = 0
        net.rel_dist_to_NDI = []

    num_samples = 0
    for i, (inputs, targets) in enumerate(dloader.train):

        # Reset optimizers.
        optimizers['forward'].zero_grad()
        if single_phase:
            optimizers['feedback'].zero_grad()
        batch_size = inputs.shape[0]

        # Make predictions.
        predictions = net.forward(inputs)

        # Inform the network whether values should be logged (once per epoch)
        if 'DFC' in network_type:
            net.save_ndi_updates = False
            if i == 0:
                net.save_ndi_updates = config.save_ndi_angle

        ### Compute loss and accuracy.
        batch_loss = loss_fn(predictions, targets)
        batch_accuracy = None
        if shared.classification:
            batch_accuracy = compute_accuracy(predictions, targets)

        ### Compute gradients and update weights.
        if compute_gradient:
            net.backward(batch_loss, targets=targets)
            if config.clip_grad_norm != -1:
                for param in extract_parameters(net, config, network_type):
                    nn.utils.clip_grad_norm_(param, config.clip_grad_norm)
                if single_phase:
                    for param in extract_parameters(net, config, network_type,\
                            params_type='feedback'):
                        nn.utils.clip_grad_norm_(param, config.clip_grad_norm)
                assert net.get_max_grad() <= config.clip_grad_norm
                
            if hasattr(config, 'use_bp_updates') and config.use_bp_updates:
                net.set_grads_to_bp(batch_loss, retain_graph=True)

            # Perform the update.
            optimizers['forward'].step()
            if single_phase:
                optimizers['feedback'].step()

        ### Compute H loss.
        if  hasattr(config, 'save_lu_loss') and config.save_lu_loss:
            epoch_loss_lu += dfc.loss_function_H(config, net, shared)

        ### Store values.
        epoch_losses.append(batch_loss.detach().cpu().numpy())
        if shared.classification:
            epoch_accs.append(batch_accuracy)

        shared.train_var.batch_idx += 1
        num_samples += batch_size

        if config.test and i == 1:
            break

    # Compute angles if needed.
    if not config.no_plots or config.save_df:
        if 'DFC' in network_type:
            dfc.save_angles(config, writer, epoch+1, net, batch_loss)
            if config.save_H_angle:
                shared.train_var.epochs_lu_angle.append(\
                    net.lu_angles[0].tolist()[-1])
            if config.save_condition_fb:
                gn_condition = net.compute_condition_two()
                shared.train_var.gn_condition.append(gn_condition.item())

    # Save results in train_var.
    shared.train_var.epochs_train_loss.append(np.mean(epoch_losses))
    if shared.classification:
        shared.train_var.epochs_train_accu.append(np.mean(epoch_accs))

    if 'DFC' in network_type:
        if config.save_lu_loss:
            shared.train_var.epochs_train_loss_lu.append(epoch_loss_lu/num_samples)
        if config.compare_with_ndi:
            shared.train_var.rel_dist_to_ndi.append(np.mean(net.rel_dist_to_ndi))

    return epoch_losses, epoch_accs

def test(config, logger, device, writer, shared, dloader, net, loss_fn,
         network_type, data_split='test'):
    """Test the network.

    Args:
        (....): See docstring of function :func:`train`.
        data_split (str): The test split to use: `test` or `validation`.

    Return:
        (....): Tuple containing:

        - **test_loss**: The average test loss.
        - **test_acc**: The average test accuracy. ``None`` for non
            classification tasks.
    """
    # Chose the correct data split.
    if data_split == 'test':
        data = dloader.test
    elif data_split == 'validation':
        data = dloader.val

    with torch.no_grad():
        test_loss = 0
        test_accu = 0 if shared.classification else None
        num_samples = 0
        for i, (inputs, targets) in enumerate(data):
            batch_size = inputs.shape[0]
            predictions = net.forward(inputs)

            ### Compute loss and accuracy.
            test_loss += batch_size * loss_fn(predictions, targets).item()
            if shared.classification:
                test_accu += batch_size * compute_accuracy(predictions, targets)      

            num_samples += batch_size
            
            if config.test and i == 1:
                break

    # For auto-encoding runs, plot some reconstructions.
    if config.dataset == 'mnist_autoencoder' and not config.no_plots:
        plt_utils.plot_auto_reconstructions(config, writer, inputs, predictions)
 
    # Because we use mean reduction and the last batch might have
    # different size, we multiply in each batch by the number of samples and
    # redivide here by the total number.
    test_loss /= num_samples
    if shared.classification:
        test_accu /= num_samples

    # Save results in train_var.
    if data_split == 'test':
        shared.train_var.epochs_test_loss.append(test_loss)
        if shared.classification:
            shared.train_var.epochs_test_accu.append(test_accu)
    elif data_split == 'validation':
        shared.train_var.epochs_val_loss.append(test_loss)
        if shared.classification:
            shared.train_var.epochs_val_accu.append(test_accu)

    return test_loss, test_accu


def compute_accuracy(predictions, labels):
    """Compute the average accuracy of the given predictions.

    Inspired by
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    Args:
        predictions (torch.Tensor): Tensor containing the output of the linear
            output layer of the network.
        labels (torch.Tensor): Tensor containing the labels of the mini-batch.

    Returns:
        (float): Average accuracy of the given predictions.
    """
    if len(labels.shape) > 1:
        # In case of one-hot-encodings, need to extract class.
        labels = labels.argmax(dim=1)

    _, pred_labels = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total
