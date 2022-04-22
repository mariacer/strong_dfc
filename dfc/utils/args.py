#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany
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
# @title          :utils/args.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
Command line argument parsing
-----------------------------

Command-line arguments common to all experiments.
"""
import argparse
from datetime import datetime
import hypnettorch.utils.cli_args as cli
from hypnettorch.utils import misc
import numpy as np
import warnings

def parse_cmd_arguments(network_type='BP', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        network_type (str): The network type to be used.
        default (optional): If ``True``, command-line arguments will be ignored
            and only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """
    curr_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dout_dir = './out/run_' + curr_date

    parser = argparse.ArgumentParser()

    #################
    ### Arguments ###
    #################

    misc_agroup = cli.miscellaneous_args(parser, big_data=False,
                                         dout_dir=dout_dir)
    miscellaneous_args(misc_agroup)
    tgroup = training_args(parser, network_type)
    optimizer_args(tgroup)
    dataset_args(parser)
    student_teacher_args(parser)
    network_args(parser)
    if network_type == 'DFC':
        optimizer_args(tgroup, suffix='_fb', dweight_decay=0.01, dmomentum=None,
                       doptimizer=None)
        dfc_args(parser)
    elif network_type == 'DFC_single_phase':
        optimizer_args(tgroup, suffix='_fb', dweight_decay=0.01, dmomentum=None,
                       doptimizer=None)
        dfc_args(parser, single_phase=True)

    ##################################
    ### Finish-up Argument Parsing ###
    ##################################
    # Including all necessary sanity checks.

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Post-process arguments.
    config = post_process_args(config, network_type)

    ### Check argument values.
    check_invalid_args_general(config, network_type)

    return config


def training_args(parser, network_type, depochs=2, dbatch_size=128,
                  dclip_grad_norm=-1):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training argument group.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        network_type (str): The type of network.
        (....): Default values for the arguments.

    Returns:
        The created argument group, in case more options should be added.
    """
    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--epochs', type=int, metavar='N', default=depochs,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N',
                        default=dbatch_size,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--clip_grad_norm', type=float, default=dclip_grad_norm,
                        help='Clip the norm of the forward and feedback weight '
                             'updates if they are bigger than the specified ' 
                             'value.')
    tgroup.add_argument('--min_acc', type=float, default=0.,
                        help='The minimal accuracy that needs to be achieved '
                             'by the end of epoch 3 for the run to not be '
                             'killed.')
    aux = ''
    if network_type == 'DFC':
        aux =' forward'
    tgroup.add_argument('--freeze_fw_weights', action='store_true',
                        help='Freeze the %sweights during training.' % aux + 
                             'Note that the epochs are still run through, '
                             'only the weight updates are not performed.')

    # Options to train only part of the network.
    if network_type == 'BP':
        tgroup.add_argument('--only_train_first_layer', action='store_true',
                            help='Only train the forward parameters of first '
                                 'layer, while freezing all other forward '
                                 'parameters to their initialization. The '
                                 'feedback parameters are all trained.')
        tgroup.add_argument('--only_train_last_layer', action='store_true',
                            help='Train only the parameters of the last layer '
                                 'and let the others stay fixed. This is used '
                                 'as a backprop baseline.')

    return tgroup

def optimizer_args(tgroup, suffix='', dlr='0.1', dmomentum=0, dweight_decay=0,
                   doptimizer='SGD', dbeta1=0.99, dbeta2=0.99, depsilon='1e-8'):
    """Optimizer options.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        suffix (str): The suffix to be added to the arguments. Can be emtpy,
            or `_fb` for the feedback weights in DFC. 
        (....): Default values for the arguments.

    Returns:
        The created argument group, in case more options should be added.
    """
    # Some helpers to make this function work for forward and feedback weights.
    name = 'forward'
    explanation = ''
    aux = ''
    if suffix == '_fb':
        name = 'feedback'
        explanation = ' for the feedback parameters'
        aux = ' If no value is provided, this value will be set to be ' + \
              'identical to that of the forward weights.'

    tgroup.add_argument('--lr%s' % suffix, type=str, default=dlr,
                        help='Learning rate of optimizer for the %s ' % name +
                             'parameters. You can either provide a single '
                             'float that will be used as lr for all the layers,'
                             'or a list of learning rates (e.g. [0.1,0.2,0.5]) '
                             'specifying a lr for each layer. The length of the'
                             ' list should be equal to num_hidden + 1. The list'
                             'may not contain spaces. Default: %(default)s.')
    tgroup.add_argument('--optimizer%s' % suffix, type=str, default=doptimizer,
                        choices=['SGD', 'RMSprop', 'Adam'],
                        help='Optimizer used for training the %s ' % name + 
                             'parameters. Default: %(default)s.')
    tgroup.add_argument('--momentum%s' % suffix, type=float, default=dmomentum,
                        help='Momentum of the SGD or RMSprop %s ' % name +
                             'optimizer.%s ' % aux + 'Default: %(default)s.')
    tgroup.add_argument('--weight_decay%s' % suffix, type=float,
                        default=dweight_decay,
                        help='Weight decay for the %s weights. ' % name +
                             'Default: %(default)s.')

    # Adam optimizer options.
    tgroup.add_argument('--adam_beta1%s' % suffix, type=float, default=dbeta1,
                        help='Adam beta1 value%s. ' % explanation +
                             'Default: %(default)s')
    tgroup.add_argument('--adam_beta2%s' % suffix, type=float, default=dbeta2,
                        help='Adam beta2 value%s. ' % explanation +
                             'Default: %(default)s')
    tgroup.add_argument('--adam_epsilon%s' % suffix, type=str, default=depsilon,
                        help='Adam epsilon value%s. ' % explanation +
                             'You can either provide a '
                             'single float that will be used for all the '
                             'layers, or a list of specifying a value for each '
                             'layer. The length of the list should be equal '
                             'to num_hidden + 1. The list may not contain '
                             'spaces. Default: %(default)s.')

    return tgroup


def dataset_args(parser, ddataset='mnist', dtarget_class_value=1):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the dataset argument group.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
    """
    dgroup = parser.add_argument_group('Dataset options')
    dgroup.add_argument('--dataset', type=str, default=ddataset,
                        choices=['mnist', 'fashion_mnist', 'mnist_autoencoder',
                                 'cifar10', 'student_teacher'],
                        help='Dataset to use for experiments. '
                             'Default: %(default)s.')
    dgroup.add_argument('--no_val_set', action='store_true',
                        help='Flag indicating that no validation set is used'
                             'during training.')
    dgroup.add_argument('--target_class_value', type=float,
                        default=dtarget_class_value,
                        help='For classification tasks, the value that the '
                             'correct class should have. Values of 1 '
                             'correspond to one-hot-encoding, and values '
                             'smaller than one correspond to soft targets. '
                             'Default: %(default)s.')

    return dgroup

def student_teacher_args(parser, dnum_train=1000, dnum_val=1000,
                         dnum_test=1000, dteacher_n_in=10, dteacher_n_out=2,
                         dteacher_size_hidden='5'):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the student-teacher argument group.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
    """
    dgroup = parser.add_argument_group('Student-teacher dataset options')
    dgroup.add_argument('--teacher_num_train', type=int, default=dnum_train,
                        help='Number of training samples used for the '
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--teacher_num_test', type=int, default=dnum_test,
                        help='Number of test samples used for the '
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--teacher_num_val', type=int, default=dnum_val,
                        help='Number of validation samples used for the'
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--teacher_n_in', type=int, metavar='N',
                        default=dteacher_n_in,
                        help='Dimensionality of the inputs. ' + 
                             'Default: %(default)s.')
    dgroup.add_argument('--teacher_n_out', type=int, metavar='N',
                        default=dteacher_n_out,
                        help='Dimensionality of the outputs. ' + 
                             'Default: %(default)s.')
    dgroup.add_argument('--teacher_size_hidden', type=str, metavar='N',
                        default=dteacher_size_hidden,
                        help='Number of units in each hidden layer of the ' +
                             '(teacher) network. Default: %(default)s.'
                             'If you provide a list, you can have layers of '
                             'different sizes (width).')
    dgroup.add_argument('--teacher_linear', action='store_true',
                        help='Flag indicating that the student-teacher dataset '
                             'is created with a linear teacher.')
    dgroup.add_argument('--data_random_seed', type=int, default=42,
                        help='The random seed for generating the data.')

def miscellaneous_args(mgroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the miscellaneous argument group.

    Args:
        mgroup: The argument group returned by method
            :func:`utils.cli_args.miscellaneous_args`.
    """
    mgroup.add_argument('--double_precision', action='store_true',
                        help='Use double precision floats (64bits) instead of '
                             '32bit floats. This slows up training, but can'
                             'help with numerical issues.')
    mgroup.add_argument('--test', action='store_true',
                        help='If active, this option will ensure that the '
                             'simulation is very fast by cutting the number '
                             'of epochs to 1, and by using only a couple of '
                             'batches. This can be used for debugging.')

    # Logging and writing.
    mgroup.add_argument('--no_plots', action='store_true',
                        help='If activated, no plots will be generated ' +
                             '(not even for the writer). This can be ' +
                             'useful to save time, e.g. for a ' +
                             'hyperparameter search.')
    mgroup.add_argument('--save_checkpoints', action='store_true',
                        help='If active, networks will be stored after '
                             'pre-training (if existing) and during training.')
    mgroup.add_argument('--checkpoint_interval', type=str, default=20,
                        help='Every how many epochs to checkpoint the models.')
    mgroup.add_argument('--pretrained_net_dir', type=str, default=None,
                        help='The path to the pre-trained network to be '
                             'loaded.')
    mgroup.add_argument('--epoch_summary_interval', type=int, default=-1,
                        help='Every how many epochs should a different '
                             'summary file be stored.')

def network_args(parser, dsize_hidden='5', dhidden_activation='linear',
                 dinitialization='xavier_normal'):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the network argument group.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        (....): Default values for the arguments.

    Returns:
        The created argument group, in case more options should be added.
    """
    sgroup = parser.add_argument_group('Network options')
    sgroup.add_argument('--size_hidden', type=str, metavar='N',
                        default=dsize_hidden,
                        help='Number of units in each hidden layer of the ' +
                             '(student) network. Default: %(default)s.'
                             'If you provide a list, you can have layers of '
                             'different sizes (width).')
    sgroup.add_argument('--hidden_activation', type=str,
                        default=dhidden_activation,
                        help='Activation function used for the hidden layers. '
                             'Default: %(default)s.')
    sgroup.add_argument('--no_bias', action='store_true',
                        help='Flag for not using biases in the network.')
    sgroup.add_argument('--initialization', type=str, default=dinitialization,
                        choices=['orthogonal', 'xavier', 'xavier_normal'],
                        help='Type of initialization that will be used for '
                             'forward (and feedback, if not DFC) weights of '
                             'the network. Default: %(default)s.')

def dfc_args(parser, dinit_fb_epochs=1,  dsigma_init=0, dextra_fb_epochs=0,
             dtarget_stepsize=0.001, depsilon_di=0.5, dtau_f=0.9,
             dtau_noise=0.8, single_phase=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add DFC arguments to the training argument group.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        (....): Default values for the arguments.
        single_phase (bool): Whether we are doing DFC with a single phase.

    Returns:
        The created argument group, in case more options should be added.
    """
    tgroup = parser.add_argument_group('Deep Feedback Control options')

    ### Dynamics options.
    tgroup.add_argument('--ss', action='store_true', # old "grad_delta_v" arg
                        help='Only update feedforward weights after steady '
                             'state.')
    tgroup.add_argument('--ssa', action='store_true', # old "ndi" argument
                         help='Only update feedforward weights after steady '
                              'state computed analytically. This speeds up '
                              'computation assuming convergence.')
    tgroup.add_argument('--noisy_dynamics', action='store_true',
                        help='Flag indicating whether the dynamics of the '
                             'system when learning the forward weights should '
                             'be corrupted slightly with white noise with std '
                             '"sigma".')
    tgroup.add_argument('--low_pass_filter_noise', action='store_true',
                        help='Low-pass filter the controller noise '
                             'when "noisy_dynamics" is active.')
    tgroup.add_argument('--tau_noise', type=float, default=dtau_noise,
                        help='Time constant of exponential filter for noise.')

    ### Output target options.
    tgroup.add_argument('--strong_feedback', action='store_true',
                        help='Whether the outputs should be clamped to their '
                             'desired values instead of being nudged towards '
                             'lower loss values.')
    tgroup.add_argument('--target_stepsize', type=float,
                        default=dtarget_stepsize,
                        help='Step size for computing the nudeged output '
                             'target based on the output gradient. '
                             'Default: %(default)s.')
    tgroup.add_argument('--error_as_loss_grad', action='store_true',
                        help='Compute the error e(t) as the gradient of the '
                             'loss with respect to the output activations, '
                             'instead of always using e(t) = r_L* - r_L(t). '
                             'This option does not change anything when using '
                             'MSE errors (since both are equivalent), but '
                             'has an impact in classification tasks. '
                             'Currently this option is only supported if the '
                             '`--strong_feedback` flag is set.')

    ### Initialization of feedback weights.
    tgroup.add_argument('--initialization_fb', type=str,
                        default='xavier_normal',
                        choices=['orthogonal', 'xavier', 'xavier_normal', 
                                 'weight_product'],
                        help='Type of initialization that will be used for '
                             'feedback weights of the network. '
                             'Default: %(default)s.')
    tgroup.add_argument('--sigma_init', type=float, default=dsigma_init,
                        help='Standard deviation of gaussian noise used to '
                             'corrupt the feedback initialization whenever '
                             '`--initialization_fb=weight_product`. '
                             'Default: %(default)s.') # old noise_K

    ### Training of the forward weights options.
    tgroup.add_argument('--normalize_lr', action='store_true',
                        help='Flag indicating that we should take the real '
                             'learning rate of the forward parameters to be:'
                             'lr=lr/target_stepsize. This makes the hpsearch'
                             'easier, as lr and target_stepsize have similar'
                             'behavior.')
    tgroup.add_argument('--learning_rule', type=str,
                        default='nonlinear_difference',
                        choices=['nonlinear_difference', 'voltage_difference'],
                        help='Type of learning rule used to train the '
                             'forward weights. Default: %(default)s.')

    ### Pre-training of feedback weights options.
    tgroup.add_argument('--init_fb_epochs', type=int, default=dinit_fb_epochs,
                        help='Number of pre-training epochs for the feedback '
                             'weights. Default: %(default)s.')
    tgroup.add_argument('--lr_fb_init', type=str, default=None,
                        help='Learning rate for pre-training feedback weights. '
                             'If ``None`` is provided, then "lr_fb" will be '
                             'used. Default: %(default)s.')
    tgroup.add_argument('--pretrain_without_controller', action='store_true',
                        help='If active, the feedback weight pretraining '
                             'will use the forward activations without '
                             'controller active as a desired target. Only has '
                             'an effect when using "DFC_single_phase", '
                             'since by default in this setting the actual '
                             'supervised labels are used as targets.')

    ### Training of feedback weights options.
    tgroup.add_argument('--freeze_fb_weights', action='store_true',
                        help='Only train the forward parameters, and keep the '
                             'feedback parameters fixed.')
    tgroup.add_argument('--freeze_fb_weights_output', action='store_true',
                        help='Freeze the feedback weights of the output layer.')
    tgroup.add_argument('--extra_fb_epochs', type=int,
                        default=dextra_fb_epochs,
                        help='Number of extra epochs that the feedback '
                             'weights will be trained for after every '
                             'epoch of forward parameter training. '
                             'Default: %(default)s.')   
    tgroup.add_argument('--scaling_fb_updates', action='store_true',
                        help='Flag indicating whether the feedback weight '
                             'gradients are layerwise scalled before the '
                             'update if performed.')

    # Convergence arguments.
    tgroup.add_argument('--include_only_converged_samples', action='store_true',
                        help='Only include samples in the mini-batch that '
                             'converged to do the update.  If noisy_dynamics'
                             'is used, this flag should be ``False``, as the '
                             'noise prevents the samples from converging '
                             'exactly.')
    tgroup.add_argument('--epsilon_di', type=float, default=depsilon_di,
                         help='Constant to check for convergence. Helps avoid '
                              'spurious minima, but setting it too low '
                              '(<0.2) can lead to slower computation (as some '
                              'error minima will not be detected as such). '
                              'Default: %(default)s.')

    def dfc_controller_args(gp, suffix='', dsigma=0.08, dtmax_di=10., 
                            dalpha_di=0.001, dk_p=0., dtime_constant_ratio=0.2,
                            dapical_time_constant=-1, ddt_di=0.1,
                            single_phase=False):
        """Add controller arguments to DFC argument group.

        Args:
            gp: The DFC parser group.
            suffix (str): The suffix to be added to the arguments. Can be emtpy,
                or `_fb` for the feedback weights. Note that in single-phase
                settings, the 'fb' options are only used for the pretraining.
                During normal training, the same dynamics as in the forward pass
                are used.
            single_phase (bool): Whether we are working with a single phase.
                If True, only the arguments `k_p_fb` and `alpha_di_fb` will
                be created, as these are important for the single-phase
                pretraining.
        """
        # In the feedback setting, many default values are set to -1.
        if suffix == '_fb':
            dtime_constant_ratio = -1
            dtmax_di = -1
            ddt_di = -1
            dapical_time_constant = -1
            dsigma = -1
            dk_p = -1
            dalpha_di = 0.5

        # Some helpers to make function work for forward and feedback weights.
        name = 'forward'
        exp = ''
        training = ''
        if suffix == '_fb':
            name = 'feedback'
            if single_phase:
                training = 'pre-'
        exp = 'when %straining the %s weights' % (training, name)


        # Temporal dynamics.
        gp.add_argument('--dt_di%s' % suffix, type=float, default=ddt_di,
                        help='Time step for Euler steps, to compute the '
                             'dynamical inversion of targets %s. ' % exp +
                             'Larger values lead to faster convergence '
                             '(specially if used with fast DI), but dt>0.5 '
                             'can lead to unstable, non-convergent '
                             'dynamics. Default: %(default)s.')
        gp.add_argument('--tmax_di%s' % suffix, type=int, default=dtmax_di,
                        help='Maximum number of iterations (timesteps) '
                             'performed in dynamical inversion of targets '
                             '%s. ' % exp + 'Default: %(default)s.')
        gp.add_argument('--inst_transmission%s' % suffix,
                        action='store_true',
                        help='Flag indicating that we assume an '
                             'instantaneous transmission between layers '
                             '%s. In this setting, in one ' % exp +
                             'simulation iteration, the basal voltage of '
                             'layer i at timestep t is based on the '
                             'forward propagation of the somatic '
                             'voltage of layer i-1 at timestep t, hence '
                             'already incorporating the feedback of the '
                             'previous layer at the current timestep.')

        # Time constants.
        gp.add_argument('--time_constant_ratio%s' % suffix, type=float,
                        default=dtime_constant_ratio,
                        help='Ratio of the time constant of the voltage '
                             'dynamics w.r.t. the controller dynamics '
                             '%s. ' % exp)
        
        # Miscellaneous.
        gp.add_argument('--sigma%s' % suffix, type=float, default=dsigma,
                        help='Standard deviation of gaussian noise used to '
                             'corrupt the layer activations during the '
                             'controller dynamics %s. ' % exp +
                             'Default: %(default)s.')
        gp.add_argument('--sigma_output%s' % suffix, type=float, default=-1,
                        help='Standard deviation of gaussian noise used to '
                             'corrupt the output activations during the '
                             'controller dynamics %s. ' % exp +
                             'Default: %(default)s.')
        gp.add_argument('--alpha_di%s' % suffix, type=float, default=dalpha_di,
                        help='Leakage gain of the feedback controller '
                             '%s. It stabilize the dynamics. ' % exp +
                             'Default: %(default)s.')
        gp.add_argument('--k_p%s' % suffix, type=float, default=dk_p,
                        help='The gain factor of the proportional control '
                             'term of the PI controller %s. ' % exp +
                             'When equal to zero, the proportional '
                             'control term is omitted and only integral '
                             'control is used. Only positive values allowed '
                             'for k_p. Default: %(default)s.')

        if not single_phase:
            # In single-phase settings, the apical dynamics are instantaneous
            # and hence require no time constants.
            gp.add_argument('--apical_time_constant%s' % suffix, type=float,
                            default=dapical_time_constant,
                            help='Time constant of the apical compartment '
                                 '%s. ' % exp +
                                 'By default it will be set equal to '
                                 '"dt_di%s", such that it results ' % suffix +
                                 'in instantaneous apical compartment '
                                 'dynamics. Default: %(default)s.')

    ### Controller options.
    tgroup.add_argument('--inst_system_dynamics', action='store_true',
                        help='Flag indicating that the system dynamics, i.e.'
                             'the dynamics of the somatic compartments, should'
                             'be approximated by their instantaneous '
                             'counterparts, i.e. equivalent to having a '
                             'time constant of lim -> 0.')
    tgroup.add_argument('--proactive_controller', action='store_true',
                        help='Use a slight variation on the forward Euler '
                             'method for simulating the controller dynamics '
                             'during the learning of the forward weights, '
                             'such that the control input u[k+1] (which '
                             'incorporates the control error e[k]) is used to '
                             'compute the apical compartment voltage v^A[k+1], '
                             'instead of using u[k].')
    tgroup.add_argument('--low_pass_filter_u', action='store_true',
                        help='Low-pass filter the controller signal u '
                             'to be used in the neuronal dynamics.')
    tgroup.add_argument('--tau_f', type=float, default=dtau_f,
                        help='Time constant of exponential filter for '
                             'voltage and controller, applied if '
                             '"low_pass_filter_u" is active.')
    # Controller for forward weight learning.
    dfc_controller_args(tgroup)
    # Controller for feedback weight learning.
    dfc_controller_args(tgroup, suffix='_fb', dsigma=-1, dalpha_di=0.5,
                        dtmax_di=-1, dk_p=None, dtime_constant_ratio=-1,
                        ddt_di=-1, single_phase=single_phase)

    ### Debugger options.
    tgroup.add_argument('--use_bp_updates', action='store_true',
                        help='Overwrite the updates with the loss gradients '
                             '(BP updates) when learning the forward weights. '
                             'The network is trained with backpropagation , '
                             'but still instantiated as DFC. This can be '
                             'useful for debugging purposes.')
    tgroup.add_argument('--use_jacobian_as_fb', action='store_true',
                        help='Use the Jacobian as the feedback weights to '
                             'control the network.')
    tgroup.add_argument('--compute_jacobian_at',
                        choices=['ss', 'average_ss', 'full_trajectory'],
                        default='ss',
                        help='How to compute the Jacobian, either at the '
                             'steady-state (ss), the average steady-state '
                             'in the last quarter of the simulartion '
                             '(average-ss) or across the entire trajectory.')
    tgroup.add_argument('--compare_with_ndi', action='store_true',
                        help='When doing dynamical control, also compute '
                             'the analytical solution in order to compare '
                             'both solutions. Causes a computational overhead.')
    tgroup.add_argument('--save_df', action='store_true',
                        help='Flag indicating that the computed angles should'
                             'be saved in dataframes for later use/plotting.')
    tgroup.add_argument('--save_fb_statistics_init', action='store_true',
                        help='Flag indicating that the statistics of the '
                             'feedback weights should be saved during the '
                             'initial training of the feedback weights.')
    ### Angle calculations.
    tgroup.add_argument('--save_ndi_angle',
                        action='store_true',
                        help='Flag indicating whether angle with the '
                             'analytical updates should be computed. Causes a '
                             'minor increase in computational load.')
    tgroup.add_argument('--save_bp_angle', action='store_true',
                        help='Flag indicating whether the BP updates and the'
                             'angle between those updates and DFC updates '
                             'should be computed and saved.')    
    tgroup.add_argument('--save_H_angle', action='store_true',
                        help='Flag indicating whether angle with the ideal '
                             'updates from loss_u should be computed. Warning, '
                             'this causes a heavy extra computational load.')
    tgroup.add_argument('--save_lu_loss', action='store_true',
                        help='Flag indicating to save help minimization loss.')
    tgroup.add_argument('--save_ratio_ff_fb', action='store_true',
                        help='Flag indicating whether ratio between the '
                             'feeforward and feedback stimulus should be '
                             'saved.')
    tgroup.add_argument('--save_condition_fb', action='store_true',
                        help='Flag indicating whether the Gauss-Newton '
                             'condition on the feedback weights should be '
                             'computed and saved.')
    return tgroup

def post_process_args(config, network_type):
    """Post process the command line arguments.

    Which kind of stuff is post-processed here for all runs?
        - reduce number of epochs if we are in a `test` setting
        - convert string into lists (for architecture, learning rates)

    Which kind of stuff is post-processed here for DFC runs?
        - adjust learning rate based on target nudging strength if needed
        - copy forward phase hyperparameters for feedback phase if not provided
        - fill in possibly missing arguments for:
            * lr_fb_init <- lr_fb
            * sigma_output <- sigma
            * sigma_output_fb <- sigma_fb
            * apical_time_constant <- dt_di
            * apical_time_constant_fb <- dt_di_fb
        - remove feedback training if Jacobian is used as feedback.
        - determine whether dataframe should be stored.

    Args:
        config: Parsed command-line arguments.
        network_type (str): The type of network.

    Returns:
        The post-processed config.
    """
    # Reduce number of epochs if we are in a `test` setting
    if config.test:
        config.epochs = 1

    # Process lists of architectures and learning rates.
    config.size_hidden = misc.str_to_ints(config.size_hidden)
    config.teacher_size_hidden = misc.str_to_ints(config.teacher_size_hidden)
    config.lr = misc.str_to_floats(config.lr)
    config.adam_epsilon = misc.str_to_floats(config.adam_epsilon)
    if len(config.lr) > 1 and len(config.adam_epsilon) == 1:
        config.adam_epsilon = config.adam_epsilon*len(config.lr)

    # DFC arguments.
    if network_type == 'DFC' or network_type == 'DFC_single_phase':

        # Reduce number of epochs if we are in a `test` setting
        if config.test:
            config.init_fb_epochs = 1
            config.extra_fb_epochs = 1

        # Process lists learning rates.
        config.lr_fb = misc.str_to_floats(config.lr_fb)
        config.adam_epsilon_fb = misc.str_to_floats(config.adam_epsilon_fb)

        # Adjust learning rate based on target nudging strength.
        if config.normalize_lr:
            config.lr = [lr/config.target_stepsize for lr in config.lr]

        # Copy forward phase hyperparameters for feedback phase if not provided
        def set_arg_equal_to_wake_phase(config, arg):
            """For backward weight phase learning arguments that are -1, we set
            the values to be equal to those for learning the forward weights."""
            fb_value = getattr(config, arg + '_fb')
            if fb_value == -1 or fb_value is None or fb_value == 'None':
                setattr(config, arg + '_fb', getattr(config, arg))
            return config
        config = set_arg_equal_to_wake_phase(config, 'optimizer')
        config = set_arg_equal_to_wake_phase(config, 'lr')
        config = set_arg_equal_to_wake_phase(config, 'momentum')
        config = set_arg_equal_to_wake_phase(config, 'k_p')
        config = set_arg_equal_to_wake_phase(config, 'dt_di')
        config = set_arg_equal_to_wake_phase(config, 'time_constant_ratio')
        config = set_arg_equal_to_wake_phase(config, 'tmax_di')
        config = set_arg_equal_to_wake_phase(config, 'inst_transmission')
        config = set_arg_equal_to_wake_phase(config, 'sigma')

        # Fill in possibly missing arguments.
        if config.lr_fb_init == None:
            config.lr_fb_init = config.lr_fb
        else:
            config.lr_fb_init = misc.str_to_floats(config.lr_fb_init)
        if config.sigma_output == -1:
            config.sigma_output = config.sigma
        if config.sigma_output_fb == -1:
            config.sigma_output_fb = config.sigma_fb
        if hasattr(config, 'apical_time_constant') and \
                                        config.apical_time_constant == -1:
            config.apical_time_constant = config.dt_di
        if hasattr(config, 'apical_time_constant_fb') and \
                                        config.apical_time_constant_fb == -1:
            config.apical_time_constant_fb = config.dt_di_fb

        # Remove feedback training if Jacobian is used as feedback.
        if config.use_jacobian_as_fb:
            config.freeze_fb_weights = True
            config.init_fb_epochs = 0
            config.extra_fb_epochs = 0

        # Determine whether dataframe should be stored.
        if config.save_ndi_angle or config.save_bp_angle or \
                config.save_H_angle or config.save_lu_loss or \
                config.save_ratio_ff_fb  or config.save_condition_fb:
           config.save_df = True

    return config

def check_dfc_time_constants(config):
    r"""Check the time constants for DFC runs.

    We need to ensure :math:`\delta_t \inf \inf \tau_v , \tau_{\epsilon} \\
    \inf \inf \tau_f \inf \inf \tau_u = 1`.

    Args:
        config: The config.

    Returns:
        The (potentially modified) config.
    """
    if config.tau_f >= 1:
        raise ValueError('"tau_f" must be smaller than "tau_u", which '
                         'is chosen to be equal to 1.')

    if config.tau_noise > config.tau_f:
        warnings.warn('"tau_noise" for exponential filtering must be '
                      'smaller than "tau_f". Setting it to "tau_f".')
        config.tau_noise = config.tau_f

    if config.time_constant_ratio > config.tau_f:
        warnings.warn('"time_constant_ratio" must be '
                      'smaller than "tau_f". Setting it to "tau_f".')
        config.time_constant_ratio = config.tau_f

    if config.time_constant_ratio_fb > config.tau_f:
        warnings.warn('"time_constant_ratio_fb" must be '
                      'smaller than "tau_f". Setting it to "tau_f".')
        config.time_constant_ratio_fb = config.tau_f

    if config.dt_di > config.time_constant_ratio:
        warnings.warn('"dt_di" must be smaller than "time_constant_ratio". '
                      'Setting it to "time_constant_ratio" to '
                      'avoid instabilities.')
        config.dt_di = config.time_constant_ratio

    if config.dt_di_fb > config.time_constant_ratio_fb:
        warnings.warn('"dt_di_fb" must be smaller than '
                      '"time_constant_ratio_fb". Setting it to '
                      '"time_constant_ratio_fb" to avoid instabilities.')
        config.dt_di_fb = config.time_constant_ratio_fb

    return config

def check_invalid_args_general(config, network_type):
    """Sanity check for command-line arguments.

    Args:
        config: Parsed command-line arguments.
        network_type (str): The type of network.
    """
    if config.clip_grad_norm != -1 and config.clip_grad_norm < 0:
        raise ValueError('"clip_grad_norm" has to be positive.')

    if len(config.lr) != 1 and len(config.lr) != len(config.size_hidden) + 1:
        raise ValueError('The learning rate provided must be either a single '
                         'value or provide one learning rate per hidden layer.')

    if len(config.lr) > 1:
        if config.optimizer == 'RMSprop':
            raise NotImplementedError('Multiple learning rates are only '
                                      'supported for SGD and Adam optimizers.')

    if len(config.adam_epsilon) != 1:
        if len(config.lr) == 1:
            raise ValueError('The use of layer-specific "adam_epsilon" '
                             'values is only supported when there are layer-'
                             'specific learning rates.')
        elif len(config.adam_epsilon) != len(config.lr):
            raise ValueError('The provided list of "adam_epsilon" values does '
                             'not have the right length.')

    if config.clip_grad_norm != -1 and config.clip_grad_norm < 0:
        warnings.warn('The provided "clip_grad_norm" value is negative. '
                      'No gradient clipping will be performed.')
        config.clip_grad_norm = -1

    if hasattr(config, 'target_class_value'):
        if config.target_class_value <= 0 or config.target_class_value > 1:
            raise ValueError('Target values for the correct class need to be '
                             'in the ]0, 1] range.')
        if config.dataset not in ['mnist', 'fashion_mnist', \
                                  'mnist_autoencoder', 'student_teacher']:
            raise NotImplementedError

    # Network-specific arguments.
    if network_type in ['BP']:
        if config.only_train_first_layer and config.only_train_last_layer:
                raise ValueError('The options "only_train_first_layer" and '
                                 '"only_train_last_layer" cannot be both ' 
                                 'active.')

    if network_type == 'DFA' and config.initialization != 'xavier_normal':
        warnings.warn('"xavier_normal" initialization is preferred for "DFA".')

    if network_type == 'DFC' or network_type == 'DFC_single_phase':

        if config.k_p < 0:
            raise ValueError('Only positive values for "k_p" are allowed.')

        if config.sigma_init < 0:
            raise ValueError('Only positive values for "sigma_init" are '
                             'allowed.')

        if config.ssa:
            if config.compare_with_ndi:
                warnings.warn('Non-dynamical inversion is already active. '
                              'Ignoring "compare_with_ndi".')
            config.compare_with_ndi = False
            if config.include_only_converged_samples:
                warnings.warn('Option "include_only_converged_samples" is '
                              'only implemented for dynamical inversion.')

        if config.freeze_fb_weights and not \
                config.initialization_fb == 'weight_product':
            warnings.warn('Feedback weights are not being trained. Setting '
                          'their initialization to "weight_product".')
            config.initialization_fb = 'weight_product'

        config = check_dfc_time_constants(config)

        if config.strong_feedback:
            if config.ssa:
                raise ValueError('The analytical inversion is not applicable '
                                 'to settings with strong feedback as the '
                                 'linearization becomes inaccurate.')

            if not config.error_as_loss_grad:
                warnings.warn('Setting "error_as_loss_grad" for the strong '
                              'feedback DFC.')
                config.error_as_loss_grad = True

            if config.target_class_value == 1:
                warnings.warn('Setting target class value to 0.99 for '
                              'strong feedback DFC.')
                config.target_class_value = 0.99

        if config.use_jacobian_as_fb:
            if config.save_condition_fb:
                warnings.warn('Feedback weights are being fixed to the network '
                              'jacobian so condition is not computed as '
                              'it always satisfied.')
                config.save_condition_fb = False
            if config.save_ndi_angle:
                warnings.warn('Feedback weights are being fixed to the network '
                              'jacobian so ndi angle is not being computed.')
                config.save_ndi_angle = False

        if config.noisy_dynamics:
            warnings.warn('Dynamics are noisy, computing the Jacobian at the '
                          'average steady-state and not at the steady-state.')
            config.compute_jacobian_at = 'average_ss'

        if config.error_as_loss_grad and (not config.strong_feedback):
            raise NotImplementedError('`--error_as_loss_grad` has not yet been '
                                      'implemented for weak feedback.')

    if network_type == 'DFC_single_phase':
        if config.ssa or config.ss:
            warnings.warn('Single phase dynamics always use noise so using '
                           'the steady-state solutions for the updates is '
                           'not recommended.')
        if config.extra_fb_epochs != 0:
            warnings.warn('The number of extra feedback epochs should ideally '
                          'be zero in single-phase experiments!')
        if not config.use_jacobian_as_fb and not config.noisy_dynamics:
            warnings.warn('Single phase learning without ideal weights '
                          'requires noisy dynamics. Setting "noisy_dynamics" '
                          ' to ``True``.')
            config.noisy_dynamics = True