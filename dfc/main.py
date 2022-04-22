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
# @title          :main.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
Main script for training networks
---------------------------------

This script is used for training networks on a certain dataset according to a
certain algorithm. It is called by the algorithm-specific scripts that call
this function with the appropriate options, namely ``run_bp.py``,
``run_dfa.py``, ``run_dfc.py`` and ``run_dfc_single_phase.py``.
"""
from argparse import Namespace
from hypnettorch.utils import torch_ckpts as ckpts
from time import time
import torch.nn as nn

from datahandlers.data_utils import generate_task
from networks.net_utils import generate_network
from networks.dfc_network_utils import train_feedback_parameters
from utils import args
from utils import math_utils as mutils
from utils import sim_utils
from utils import train_utils as tutils
from utils.optimizer_utils import get_optimizers

def run(network_type='BP'):
    """Run the experiment.

    This script does the following:
        - parse command-line arguments
        - initialize loggers and writers
        - create datahandler
        - create network
        - train network

    Args:
        network_type (str): The type of network.

    Returns:
        summary (dict): The results summary.
    """
    ### Start simulation.
    script_start = time()
    config = args.parse_cmd_arguments(network_type=network_type)
    device, writer, logger = sim_utils.setup_environment(config)

    ### Simple struct, that is used to share data among functions.
    shared = Namespace()
    if config.dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        shared.classification = True
    elif config.dataset in ['mnist_autoencoder', 'student_teacher']:
        shared.classification = False

    ### Create the task.
    dloader = generate_task(config, logger, device)

    ### Create the networks.
    net = generate_network(config, dloader, device, network_type, logger=logger)

    ### Create the optimizers.
    optimizers = get_optimizers(config, net, network_type=network_type,
                                logger=logger)

    ### Initialize the performance measures that are tracked during training.
    shared = sim_utils.setup_summary_dict(config, shared, network_type)

    ### Define the loss function to be used.
    if shared.classification:
        loss_fn = mutils.cross_entropy_fn()
    else:
        loss_fn = nn.MSELoss()

    ### If necessary, pre-train feedback weights.
    if 'DFC' in network_type and config.pretrained_net_dir is None:
        train_feedback_parameters(config, logger, writer, device, dloader, net,
                                  optimizers, shared, loss_fn, pretraining=True)

    ### If required, load pre-trained model.
    if config.pretrained_net_dir is not None:
        cpt = ckpts.load_checkpoint(config.pretrained_net_dir, net,
                                    device=device)
        logger.info('Loaded %s network.' % cpt['net_state'])

    ### Train the network.
    shared = tutils.train(config, logger, device, writer, dloader, net,
                          optimizers, shared, network_type, loss_fn)

    ### Finish the simulation.
    writer.close()
    sim_utils.save_summary_dict(config, shared)
    logger.info('Program finished successfully in %.2f sec.'
                % (time()-script_start))

    if not config.no_plots:
        print('\nTensorboard plots: ')
        print('tensorboard --logdir=%s'%config.out_dir)

    return shared.summary

if __name__=='__main__':
    run()
