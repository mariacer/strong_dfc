Deep Feedback Control code
==========================

Running experiments
-------------------

Experiments have to be run for the different algorithms from the corresponding executable files. Datasets can be chosen through the command line option ``--dataset``.

Deep Feedback Control runs
^^^^^^^^^^^^^^^^^^^^^^^^^^

For running standard DFC you can run:

.. code-block:: console

    $ python3 run_dfc.py

Important options that you might want to use:

- ``--strong_feedback``: Uses strong feedback influences instead of nudging the output towards lower loss.
- ``--ss``: Only update forward weights at the steady state.
- ``--ssa``: Only update forward weights at the steady state computed analytically.
- ``--noisy_dynamics``: Add noise to the forward weight learning.

For all remaining options, please refer to ``utils/args.py`` or type:

.. code-block:: console

    $ python3 run_dfc.py --help
    
For running single-phase DFC (the forward and backward weights are trained simultaneously in a single phase) execute:

.. code-block:: console

    $ python3 run_dfc_single_phase.py

Baselines
^^^^^^^^^

Backpropagation experiments can be run as follows:

.. code-block:: console

    $ python3 run_bp.py

Direct Feedback Alignment experiments can be run as follows:

.. code-block:: console

    $ python3 run_dfa.py
