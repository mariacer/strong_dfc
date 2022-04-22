#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
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
# @title          :utils/sim_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for simulations
--------------------------------

A collection of helper functions for simulations to keep other scripts clean.
"""
from argparse import Namespace
from hypnettorch.utils import sim_utils as htsutils
import numpy as np
import os
import pickle
import sys
import torch

def setup_environment(config):
    """Set up the environment.

    This function should be called at the beginning of a simulation script
    (right after the command-line arguments have been parsed). The setup will
    incorporate:

        - creates the output folder
        - initializes the logger
        - makes computation deterministic if necessary
        - selects the torch device
        - creates Tensorboard writer
        - stores the command line arguments

    Args:
        config: Command-line arguments.

    Returns:
        (tuple): Tuple containing:

        - **device**: Torch device to be used.
        - **writer**: Tensorboard writer. Note, you still have to close the
          writer manually!
        - **logger**: Console (and file) logger.
    """
    device, writer, logger = htsutils.setup_environment(config,
                                    logger_name='dfc_logger')
    if config.double_precision:
        torch.set_default_dtype(torch.float64)

    # Backup command line arguments.
    backup_cli_command(config)

    # Generate plots folder if needed.
    if not config.no_plots:
        os.makedirs(os.path.join(config.out_dir, 'figures'))

    return device, writer, logger

def backup_cli_command(config):
    """Write the curret CLI call into a script.

    This will make it very easy to reproduce a run, by just copying the call
    from the script in the output folder. However, this call might be ambiguous
    in case default values have changed. In contrast, all default values are
    backed up in the file ``config.json``.

    Args:
        config: Command-line arguments.
    """
    script_name = sys.argv[0]
    run_args = sys.argv[1:]
    command = 'python3 ' + script_name
    # FIXME Call reconstruction fails if user passed strings with white spaces.
    for arg in run_args:
        command += ' ' + arg

    fn_script = os.path.join(config.out_dir, 'cli_call.sh')

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# The user invoked CLI call that caused the creation of\n')
        f.write('# this output folder.\n')
        f.write(command)

def get_summary_keys(classification):
    """Get the required summary keys for the experiment.

    Args:
        classification (boolean): Whether it is a classification task.

    Returns:
        The list of keys.
    """
    summary_keys = [# Average training loss on last epoch.
                'loss_train_last',
                # Best average training loss across all epochs.
                'loss_train_best',
                # Average testing loss on last epoch.
                'loss_test_last',
                # Best average testing loss across all epochs.
                'loss_test_best',
                # Average validation loss on last epoch.
                'loss_val_last',
                # Best average validation loss across all epochs.
                'loss_val_best',
                # Epoch with the best validation loss.
                'epoch_best_loss',
                # Train loss on best validation epoch.
                'loss_train_val_best',
                # Test loss on best validation epoch.
                'loss_test_val_best']
    if classification:
        classif_keys = [# Average training accuracy on last epoch.
                        'acc_train_last',
                        # Best average training accuracy across all epochs.
                        'acc_train_best',
                        # Average testing accuracy on last epoch.
                        'acc_test_last',
                        # Best average testing accuracy across all epochs.
                        'acc_test_best',
                        # Average validation accuracy on last epoch.
                        'acc_val_last',
                        # Best average validation accuracy across all epochs.
                        'acc_val_best',
                        # Epoch with the best validation accuracy.
                        'epoch_best_acc',
                        # Train loss on best validation epoch.
                        'acc_train_val_best',
                        # Test loss on best validation epoch.
                        'acc_test_val_best',
                    ]
        summary_keys.extend(classif_keys)
    # Average time taken by an epoch.
    summary_keys.append('avg_time_per_epoch')
    # Whether the simulation finished.
    summary_keys.append('finished')

    return summary_keys

def setup_summary_dict(config, shared, network_type):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This function adds this summary dictionary to the shared object.

    Args:
        config: Command-line arguments
        shared: A structure to share important run information.
        network_type (str): The type of network.

    Returns:
        The shared structure with the empty dictionary added.
    """
    summary = dict()

    summary_keys = get_summary_keys(classification=shared.classification)
    for k in summary_keys:
        if k == 'finished':
            summary[k] = 0
        else:
            summary[k] = -1

    shared.summary = summary

    # Store this summary file in the results folder already.
    save_summary_dict(config, shared)

    # Create holder for important training quantities to keep track of.
    shared.train_var = initialize_train_var_holder(config, network_type)

    return shared

def save_summary_dict(config, shared, epoch=None):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
        epoch (int): The current epoch. If provided, it will generate another
            performance summary file that will not be overwritten.
    """
    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    suffix = ''
    out_dir = config.out_dir
    if epoch is not None:
        suffix = '_epoch' + str(epoch)
        out_dir = os.path.join(out_dir, 'performance_overviews')

    summary_fn = 'performance_overview%s' % suffix
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, summary_fn + '.txt'), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, list_to_str(v)))
            elif isinstance(v, float) or isinstance(v, np.float32):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))

    # We also store a pickle file with the summary.
    with open(os.path.join(\
                out_dir, summary_fn + '.pickle'), 'wb') as handle:
        pickle.dump(shared.summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def initialize_train_var_holder(config, network_type):
    """Initialize a holder for all training results collected.

    Args:
        config: The command line arguments.
        network_type (str): The type of network.

    Returns:
        (Namespace): The variable holder.
    """
    train_var = Namespace()

    train_var.epochs_time = []
    train_var.epochs_train_loss, train_var.epochs_train_accu = [] ,[]
    train_var.epochs_test_loss, train_var.epochs_test_accu = [] ,[]
    if not config.no_val_set:
        train_var.epochs_val_loss, train_var.epochs_val_accu = [] ,[]
    train_var.batch_idx = 1

    if 'DFC' in network_type:
        train_var.batch_idx_fb_init = 1
        if config.compare_with_ndi:
            train_var.rel_dist_to_ndi = []
        if config.save_lu_loss:
            train_var.epochs_train_loss_lu = []
        if config.save_H_angle:
            train_var.epochs_lu_angle = []
        if config.save_condition_fb:
            train_var.gn_condition = []
            train_var.gn_condition_init = []
        train_var.batch_idx_fb = 1

    return train_var

def update_summary_info(config, shared, network_type):
    """Write information of training into the summary dictionary.

    Write information into summary and save new summary.

    Args:
        config: The command line arguments.
        shared: The shared object.
        network_type (str): The type of network.

    Returns:
        shared: The shared object with the summary updated.
    """
    tv = shared.train_var
    summary = shared.summary
    summary['avg_time_per_epoch'] = np.mean(tv.epochs_time)
    summary['loss_train_last'] = tv.epochs_train_loss[-1]
    summary['loss_train_best'] = np.min(tv.epochs_train_loss)
    summary['loss_test_last'] = tv.epochs_test_loss[-1]
    summary['loss_test_best'] = np.min(tv.epochs_test_loss)
    if not config.no_val_set:
        summary['loss_val_last'] = tv.epochs_val_loss[-1]
        summary['loss_val_best'] = np.min(tv.epochs_val_loss)
        best_e = np.argmin(tv.epochs_val_loss)
        summary['epoch_best_loss'] = best_e
        summary['loss_test_val_best'] = tv.epochs_test_loss[best_e]
        summary['loss_train_val_best'] = tv.epochs_train_loss[best_e]
    if shared.classification:
        summary['acc_train_last'] = tv.epochs_train_accu[-1]
        summary['acc_train_best'] = np.max(tv.epochs_train_accu)
        summary['acc_test_last'] = tv.epochs_test_accu[-1]
        summary['acc_test_best'] = np.max(tv.epochs_test_accu)
        if not config.no_val_set:
            summary['acc_val_last'] = tv.epochs_val_accu[-1]
            summary['acc_val_best'] = np.max(tv.epochs_val_accu)
            best_e = np.argmax(tv.epochs_val_accu)
            summary['epoch_best_acc'] = best_e
            summary['acc_test_val_best'] = tv.epochs_test_accu[best_e]
            summary['acc_train_val_best'] = tv.epochs_train_accu[best_e]

    if 'DFC' in network_type:
        if config.compare_with_ndi:
            summary['dist_to_ndi'] = tv.rel_dist_to_ndi
        if config.save_lu_loss:
            summary['loss_lu_train_last'] = tv.epochs_train_loss_lu[-1]
            summary['loss_lu_train_best'] = np.min(tv.epochs_train_loss_lu)
            summary['loss_lu_train'] = tv.epochs_train_loss_lu
        if config.save_condition_fb:
            summary['gn_condition_init'] = tv.gn_condition_init[-1]
            summary['gn_condition'] = np.mean(tv.gn_condition)
        if config.save_H_angle:
            summary['lu_angle_last'] = tv.epochs_lu_angle[-1]
            summary['lu_angle_best'] = np.min(tv.epochs_lu_angle)
            summary['lu_angle'] = tv.epochs_lu_angle

    shared.summary = summary
    save_summary_dict(config, shared)

    return shared

def add_summary_to_writer(config, shared, writer, epoch):
    """Write information of training into the writer.

    Args:
        config: The command line arguments.
        shared: The shared object.
        train_var: The training results.
        epoch (int): The current epoch.
    """
    e = epoch
    train_var = shared.train_var
    if not config.no_plots:
        writer.add_scalar('time', train_var.epochs_time[-1], e)

        # Training information.
        writer.add_scalar('train/loss', train_var.epochs_train_loss[-1], e)
        if shared.classification:
            writer.add_scalar('train/acc', train_var.epochs_train_accu[-1], e)
        if hasattr(config, 'save_lu_loss') and config.save_lu_loss:
            writer.add_scalar('train/loss_H',
                                        train_var.epochs_train_loss_lu[-1], e)
        if hasattr(config, 'save_H_angle') and config.save_H_angle:
            writer.add_scalar('train/angle_H', train_var.epochs_lu_angle[-1], e)

        # Testing information.
        writer.add_scalar('test/loss', train_var.epochs_test_loss[-1], e)
        if shared.classification:
            writer.add_scalar('test/acc', train_var.epochs_test_accu[-1], e)

        # Validation information.
        if not config.no_val_set:
            writer.add_scalar('val/loss', train_var.epochs_val_loss[-1], e)
            if shared.classification:
                writer.add_scalar('val/acc', train_var.epochs_val_accu[-1], e)

def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.

    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.

    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def log_stats_to_writer(config, writer, step, net, init=False):
    """Save logs and plots for the current mini-batch on tensorboardX.

    Saves information about feedback and forward weights.

    Args:
        config (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network.
        init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
    """
    log_weights = 'forward'
    # If feedback weights exist and they are learn, log the feedback.
    if hasattr(net, 'feedback_params') and not \
        (config.use_jacobian_as_fb or config.freeze_fb_weights):
        log_weights = 'both'
    # If we are at pre-training, don't log the forward.
    if init:
        log_weights = 'feedback' if log_weights == 'both' else None

    prefix = ''
    if init:
        prefix = 'pretraining/'

    if not config.no_plots:
        net.save_logs(writer, step, log_weights=log_weights, prefix=prefix)
