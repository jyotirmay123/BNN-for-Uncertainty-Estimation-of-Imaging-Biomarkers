from quicknat import QuickNat
from ncm import modules as sm
import torch
import torch.nn as nn


class MultiInputQuickNat(QuickNat):
    """
    A PyTorch implementation of multi input variant of QuickNAT

    """

    def __init__(self, params):
        """
          :param params: {'num_channels':1,
                          'num_filters':64,
                          'kernel_h':5,
                          'kernel_w':5,
                          'stride_conv':1,
                          'pool':2,
                          'stride_pool':2,
                          'num_classes':28
                          'se_block': False,
                          'drop_out':0.2}
          """
        super(MultiInputQuickNat, self).__init__(params)

        params['broadcasting_needed'] = True
        self.ConcatBlock = sm.ConcatBlock(params)
        self.no_of_samples = params['latent_variables']
        self.conv1 = nn.Conv2d(self.no_of_samples, self.no_of_samples, 1)
        params['num_channels'] = params['num_channels'] + self.no_of_samples
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input, another_input=None):
        """
          :param input: X
          :param another_input: Sample generated from EncoderNet,
          :return: probabiliy map
          """
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)

        bn = self.bottleneck.forward(e4)

        d4 = self.decode4.forward(bn, out4, ind4)
        d3 = self.decode1.forward(d4, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)

        if another_input is not None:
            another_input = another_input.unsqueeze(dim=2).unsqueeze(dim=2)
            c1 = self.conv1(another_input)
            c2 = self.conv1(c1)
            expanded_inp = c2.expand(-1, self.no_of_samples, d1.size(2), d1.size(3))
            d1 = torch.cat((d1, expanded_inp), 1)

        prob = self.classifier.forward(d1)

        return prob
