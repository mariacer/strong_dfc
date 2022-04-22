Reproducing experimental results from "Minimizing Control for Credit Assignment with Strong Feedback"
=====================================================================================================

.. content-inclusion-marker-do-not-remove

MNIST
-----

Backpropagation
^^^^^^^^^^^^^^^

The following run with **40 epochs** achieves around 98.17% test accuracy.

.. code-block:: console

    $ python3 run_bp.py --epochs=40 --clip_grad_norm=1.0 --lr="0.002" --optimizer="Adam" --adam_beta1=0.9 --adam_beta2=0.999 --adam_epsilon="0.002" --dataset="mnist"  --size_hidden="256,256,256" --hidden_activation="tanh" --initialization="xavier_normal"

DFC
^^^

The following run with **40 epochs** achieves around 97.80% test accuracy.

.. code-block:: console

    $ python3 run_dfc.py --double_precision --epochs=40 --batch_size=32 --clip_grad_norm=1.0 --lr=0.0010924329072635147 --optimizer=Adam --adam_beta1=0.9 --adam_epsilon=5.83238643406511e-07 --dataset=mnist --hidden_activation=tanh --initialization=xavier_normal --lr_fb=4.348352883654314e-05 --optimizer_fb=Adam --weight_decay_fb=0.1 --adam_epsilon_fb=3.355535527361205e-05 --target_stepsize=0.013812748252649061 --initialization_fb=xavier_normal --learning_rule=nonlinear_difference --init_fb_epochs=20 --lr_fb_init=0.0001524121535306385 --proactive_controller --dt_di=0.02 --tmax_di=500 --inst_transmission --time_constant_ratio=0.2 --sigma=0.035483751742735006 --alpha_di=0.0017723664900917734 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.0011421804343878519 --tmax_di_fb=44 --inst_transmission_fb --time_constant_ratio_fb=0.005870467170683501 --alpha_di_fb=0.39857902933813705 --k_p_fb=0.0713515876863314 --apical_time_constant_fb=0.1404664564483908 --size_hidden=256,256,256

Strong-DFC
^^^^^^^^^^

The following run with **40 epochs** achieves around 97.72% test accuracy.

.. code-block:: console

    $ python3 run_dfc_single_phase.py --double_precision --epochs=40 --clip_grad_norm=1.0 --lr=0.0003095880739373081 --optimizer="Adam" --adam_beta1=0.9 --adam_epsilon=1.6288767089312233e-05 --dataset="mnist" --target_class_value=0.99 --size_hidden="256,256,256" --hidden_activation="tanh" --initialization="xavier_normal" --lr_fb=2.0091405897656825e-08 --optimizer_fb="Adam" --weight_decay_fb=0.5 --adam_beta2_fb=0.999 --adam_epsilon_fb=1.1023320392297719e-11 --tau_noise=0.07 --strong_feedback --error_as_loss_grad --initialization_fb="xavier_normal" --learning_rule="nonlinear_difference" --init_fb_epochs=1 --lr_fb_init=4.5160347135162454e-07 --scaling_fb_updates --proactive_controller --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.009268288409004458 --sigma=0.004278964325349612 --sigma_output=0.029713880264874388 --alpha_di=7.608673370995652e-06 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.0012167769308505587 --tmax_di_fb=500 --inst_transmission_fb --time_constant_ratio_fb=0.014441072404716454 --sigma_fb=2.760795719756352e-05 --sigma_output_fb=1.5149030252289032e-05 --alpha_di_fb=0.016257243484558315 --k_p_fb=0.0028004742554235116 --save_df --save_condition_fb --noisy_dynamics

Strong-DFC (2-phase)
^^^^^^^^^^^^^^^^^^^^

The following run with **40 epochs** achieves around 97.75% test accuracy.

.. code-block:: console

    $ python3 run_dfc.py --double_precision --epochs=40 --clip_grad_norm=1.0 --lr=0.0006560184719951886 --optimizer="Adam" --adam_beta1=0.9 --adam_epsilon=0.00023225688276019436 --dataset="mnist" --target_class_value=0.9 --size_hidden="256,256,256" --hidden_activation="tanh" --initialization="xavier_normal" --lr_fb=9.776610079269425e-06 --optimizer_fb="Adam" --weight_decay_fb=1e-05 --adam_beta1_fb=0.999 --adam_beta2_fb=0.9 --adam_epsilon_fb=0.0019707791920650247 --ss --tau_noise=0.05 --target_stepsize=0.5 --error_as_loss_grad --initialization_fb="xavier_normal" --learning_rule="nonlinear_difference" --init_fb_epochs=1 --lr_fb_init=0.0002730319014715041 --proactive_controller --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.03553062335953924 --sigma=0.15 --alpha_di=0.008082122141749502 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.0031943783402810927 --tmax_di_fb=500 --inst_transmission_fb --time_constant_ratio_fb=0.0072383695016716215 --alpha_di_fb=0.9944618528446518 --k_p_fb=0.0691564739128701 --apical_time_constant_fb=0.12961120842685991 --strong_feedback

Fashion-MNIST
-------------

Backpropagation
^^^^^^^^^^^^^^^

The following run with **40 epochs** achieves 89.40% test accuracy.

.. code-block:: console

    $ python3 run_bp.py --epochs=40 --clip_grad_norm=1.0 --lr="0.0004" --optimizer="Adam" --adam_beta1=0.9  --adam_beta2=0.999 --adam_epsilon="0.0002" --dataset="fashion_mnist" --size_hidden="256,256,256" --hidden_activation="tanh" --initialization="xavier_normal"

DFC
^^^

The following run with **40 epochs** achieves around 88.70% test accuracy.

.. code-block:: console

    $ python3 run_dfc.py --double_precision --epochs=40 --clip_grad_norm=1.0 --lr=0.0010672718162774038 --optimizer="Adam" --adam_beta2=0.999 --adam_epsilon=3.210886729268312e-07 --dataset="fashion_mnist" --target_class_value=0.999 --size_hidden="256,256,256" --hidden_activation="tanh" --initialization="xavier_normal" --lr_fb=4.3780877225194805e-05 --optimizer_fb="Adam" --weight_decay_fb=0.001 --adam_epsilon_fb=1.0433715362737473e-08 --target_stepsize=0.019957011262493028 --initialization_fb="xavier_normal" --learning_rule="nonlinear_difference" --init_fb_epochs=20 --lr_fb_init=0.00034156927968192547 --proactive_controller --dt_di=0.02 --tmax_di=500 --inst_transmission --sigma=0.001 --alpha_di=0.000451125353657906 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.004294669581123954 --tmax_di_fb=54 --inst_transmission_fb --time_constant_ratio_fb=0.009266592158186437 --sigma_fb=0.001281308789330547 --alpha_di_fb=0.611510021981854 --k_p_fb=0.010409666247193437 --apical_time_constant_fb=0.4318893018524684

Strong-DFC
^^^^^^^^^^

The following run with **40 epochs** achieves around 87.26% test accuracy.

.. code-block:: console

    $ python3 run_dfc_single_phase.py --double_precision --epochs=40 --clip_grad_norm=1.0 --lr=0.0009531990715839369 --optimizer=Adam --adam_beta2=0.999 --adam_epsilon=9.710433697812103e-08 --dataset=fashion_mnist --target_class_value=0.99 --size_hidden=256,256,256 --hidden_activation=tanh --initialization=xavier_normal --lr_fb=2.286529865070366e-08 --optimizer_fb=Adam --weight_decay_fb=0 --adam_beta2_fb=0.999 --adam_epsilon_fb=1.8382415746971348e-11 --tau_noise=0.05 --strong_feedback --error_as_loss_grad --initialization_fb=xavier_normal --learning_rule=nonlinear_difference --init_fb_epochs=20 --lr_fb_init=3.431924248095537e-07 --scaling_fb_updates --proactive_controller --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.012867645074062533 --sigma=0.0033626248925544607 --sigma_output=0.040920898808660004 --alpha_di=7.6370562381715e-05 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.0006919071551575181 --tmax_di_fb=500 --inst_transmission_fb --time_constant_ratio_fb=0.0006919071551575181 --sigma_fb=3.885866343508479e-06 --sigma_output_fb=1.9337256383926155e-05 --alpha_di_fb=0.6266762258519998 --k_p_fb=0.02488635344325425 --save_df --save_condition_fb --noisy_dynamics

Strong-DFC (2-phase)
^^^^^^^^^^^^^^^^^^^^

The following run with **40 epochs** achieves around 87.22% test accuracy.

.. code-block:: console

    $ python3 run_dfc.py --double_precision --epochs=40 --clip_grad_norm=1.0 --lr=0.0011969828175078612 --optimizer=Adam --adam_beta1=0.9 --adam_beta2=0.9 --adam_epsilon=2.9646405819945323e-07 --dataset=fashion_mnist --target_class_value=0.999 --size_hidden=256,256,256 --hidden_activation=tanh --initialization=xavier_normal --lr_fb=1.587623302238223e-05 --optimizer_fb=Adam --weight_decay_fb=0.1 --adam_beta2_fb=0.999 --adam_epsilon_fb=8.46521372179368e-08 --ss --tau_noise=0.05 --target_stepsize=0.5 --error_as_loss_grad --initialization_fb=xavier_normal --learning_rule=nonlinear_difference --init_fb_epochs=1 --lr_fb_init=0.00042756758315276166 --extra_fb_epochs=2 --proactive_controller --tau_f=0.5 --dt_di=0.001 --tmax_di=500 --inst_transmission --time_constant_ratio=0.007427384207234133 --sigma=0.15 --alpha_di=0.0013123372965380246 --k_p=2.0 --apical_time_constant=-1 --dt_di_fb=0.0019841648377832226 --tmax_di_fb=500 --inst_transmission_fb --time_constant_ratio_fb=0.009034403214049455 --alpha_di_fb=0.41918471180494776 --k_p_fb=0.09906907227394046 --apical_time_constant_fb=0.22603240790914256 --strong_feedback
