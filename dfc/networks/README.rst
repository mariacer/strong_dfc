Networks
********

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

Implementation of different types of networks. The common network structure is specified in :mod:`networks.network_interface`. This is an abstract method from which specific network types should inherit. It uses an abstract layer that is specified in :mod:`networks.layer_interface`.

For example, a backpropagation network inherits from these two abstract classes and uses :mod:`networks.bp_networks.bp_layer` within :mod:`networks.bp_networks.bp_network`.

Important functions for performing credit assignment, such as classes for performing non-linear operations within layers with a specific forward and backward pass, are specified in :mod:`networks.credit_assignment_functions`.