Deep Feedback Control code
==========================

Running experiments
-------------------

Deep Feedback Control runs
^^^^^^^^^^^^^^^^^^^^^^^^^^

For running standard DFC you can run:

.. code-block:: console

    $ python3 run_dfc.py

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

Hyperparameter searches
-----------------------

For instructions on how to run hyperparameter searches refer to the README inside the ``hpsearch`` subfolder.
