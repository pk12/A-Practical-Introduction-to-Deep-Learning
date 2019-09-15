# -*- coding: utf-8 -*-

from __future__ import print_function


"""
Created on Tue Jul 16 21:28:37 2019

@author: Oppai
"""

"""Ladder Network
* based on Semi-Supervised Ladder Network
* the neural network is MLP network which is enough for MNIST 
"""





import torch
from Encoder import StackedEncoders
from Decoder import StackedDecoders



class Ladder(torch.nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes, encoder_activations,
                 encoder_train_bn_scaling, noise_std, use_cuda):
        super(Ladder, self).__init__()
        self.use_cuda = use_cuda
        decoder_in = encoder_sizes[-1]
        encoder_in = decoder_sizes[-1]
        self.se = StackedEncoders(encoder_in, encoder_sizes, encoder_activations,
                                  encoder_train_bn_scaling, noise_std, use_cuda)
        self.de = StackedDecoders(decoder_in, decoder_sizes, encoder_in, use_cuda)
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)

    def forward_encoders_clean(self, data): 
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)