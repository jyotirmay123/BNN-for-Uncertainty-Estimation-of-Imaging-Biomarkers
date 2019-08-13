"""Quicknat architecture"""
import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import bayesian_modules as bm


class BrainNet(nn.Module):
    """
    A PyTorch implementation of Brain_Net

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
        super(BrainNet, self).__init__()

        self.encode1 = bm.EncoderBayesianBlock(params)
        params['num_channels'] = 64
        self.encode2 = bm.EncoderBayesianBlock(params)
        self.encode3 = bm.EncoderBayesianBlock(params)
        self.encode4 = bm.EncoderBayesianBlock(params)
        self.bottleneck = bm.EncoderBayesianBlock(params)
        params['num_channels'] = 128
        self.decode1 = bm.DecoderBayesianBlock(params)
        self.decode2 = bm.DecoderBayesianBlock(params)
        self.decode3 = bm.DecoderBayesianBlock(params)
        self.decode4 = bm.DecoderBayesianBlock(params)
        params['num_channels'] = 64
        self.classifier = bm.ClassifierBayesianBlock(params)

    def forward(self, input, switch=False):
        """

        :param input: X
        :return: probabiliy map
        """
        e1, out1, ind1, kl_e1 = self.encode1.forward(input, switch=switch)
        e2, out2, ind2, kl_e2 = self.encode2.forward(e1, switch=switch)
        e3, out3, ind3, kl_e3 = self.encode3.forward(e2, switch=switch)
        e4, out4, ind4, kl_e4 = self.encode4.forward(e3, switch=switch)

        _, bn, _, kl_bn = self.bottleneck.forward(e4, pool_required=False, switch=switch)

        d4, kl_d4 = self.decode4.forward(bn, out4, ind4, switch=switch)
        d3, kl_d3 = self.decode1.forward(d4, out3, ind3, switch=switch)
        d2, kl_d2 = self.decode2.forward(d3, out2, ind2, switch=switch)
        d1, kl_d1 = self.decode3.forward(d2, out1, ind1, switch=switch)
        prob, kl_cls = self.classifier.forward(d1, switch=switch)

        if switch is True:
            kl_loss = 0.1 * (kl_e1 + kl_e2 + kl_e3 + kl_e4 + kl_bn + kl_d4 + kl_d3 + kl_d2 + kl_d1 + kl_cls)
            return prob, kl_loss
        else:
            return prob, torch.tensor([0.0])

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out, _ = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction
