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
# @title          :utils/plt_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :28/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for plotting
-----------------------------

A collection of functions for plotting.
"""
import matplotlib.pyplot as plt
import os
import torch

def plot_auto_reconstructions(config, writer, inputs, predictions,
                              num_images=5):
    """Plot MNIST autoencoding reconstructions.

    Args:
        config: The config.
        writer: The tensorboard writer.
        inputs (torch.Tensor): The inputs.
        predictions (torch.Tensor): The predictions.
        num_images (int): The number of images to print.
    """
    batch_size = inputs.shape[0] # last batch might be smaller
    if batch_size < num_images:
        num_images = batch_size
    mnist_dims = (batch_size, 1, 28, 28)

    # Reshape the images.
    input_images = torch.reshape(inputs, mnist_dims)[:num_images]
    rectr_images = torch.reshape(predictions, mnist_dims)[:num_images]
    path = os.path.join(config.out_dir, 'autoencoder_images')

    for i in range(num_images):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(input_images[i, 0, :, :].detach().cpu().numpy())
        ax1.set_title('input')
        ax2.imshow(rectr_images[i, 0, :, :].detach().cpu().numpy())
        ax2.set_title('reconstruction')
        tag = os.path.join(path, 'reconstruction%i.png' % i)
        writer.add_figure(tag=tag, figure=fig, global_step=None)