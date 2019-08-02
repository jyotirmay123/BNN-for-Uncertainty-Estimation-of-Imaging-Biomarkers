from nn_common_modules import modules as sm
import torch.nn as nn
import torch
from squeeze_and_excitation import squeeze_and_excitation as se


class EncoderNet(nn.Module):
    """
    A simple Encoder Network for Conditional-variational auto encoding.
    """

    def __init__(self, params):
        super(EncoderNet, self).__init__()

        params['broadcasting_needed'] = False
        self.concat = sm.ConcatBlock(params)
        self.encode1 = sm.SDnetEncoderBlock(params, se_block_type=se.SELayer.NONE)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params, se_block_type=se.SELayer.NONE)
        self.encode3 = sm.SDnetEncoderBlock(params, se_block_type=se.SELayer.NONE)
        self.encode4 = sm.SDnetEncoderBlock(params, se_block_type=se.SELayer.NONE)

        self.fc_mu = nn.Linear(64, params['latent_variables'])
        self.fc_sigma = nn.Linear(64, params['latent_variables'])

    def reparameterize(self, mu, logvar):
        """

        :param mu: mean
        :param logvar: variance
        :return: randomly generated sample
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inp, condition_on=None):
        """

        :param inp: X (Input image)
        :param condition_on: y (Ground Truth)
        :return: mu, sigma (C-VAE outputs)
        """

        if condition_on is not None:
            inp = torch.cat([inp, condition_on.unsqueeze(dim=1)], dim=1)

        e1, out1, ind1 = self.encode1.forward(inp)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)
        e5 = torch.mean(e4.view(e4.size(0), e4.size(1), -1), dim=2)
        mu, sigma = self.fc_mu(e5), self.fc_sigma(e5)
        return mu, sigma

    @property
    def is_cuda(self):
        """
        Check if saved_models parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda
