""" Hierarchical Quicknat architecture"""
import numpy as np
import torch
import torch.nn as nn
from projects.hierarchical_quicknat.parts.multi_input_residual_quicknat import \
    MultiInputResidualQuickNat as PriorQuickNat
from projects.hierarchical_quicknat.parts.multi_input_residual_posterior_quickant import \
    MultiInputResidualPosteriorQuickNat as PosteriorQuickNat


class HQuicknat(nn.Module):
    """
    A PyTorch implementation of Hierarchical Quicknat

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
        super(HQuicknat, self).__init__()
        self.priorNat = PriorQuickNat(params)
        self.posteriorNat = PosteriorQuickNat(params)

        self.is_training = True
        self.priorNat.is_training = self.is_training

    def set_is_training(self, is_training):
        self.is_training = is_training
        self.priorNat.is_training = is_training

    def forward(self, inp, ground_truth=None):
        # if not self.is_training:
        #     self.priorNat.enable_test_dropout()

        if ground_truth is not None:
            self.posteriorNat.forward(inp, ground_truth)
            posterior_samples = self.posteriorNat.get_samples()
        else:
            posterior_samples = None
            self.set_is_training(False)

        out = self.priorNat.forward(inp, posterior_samples)

        # if not self.is_training:
        #     prior_weights = self.priorNat.get_prior_weights_for_posterior_samplings()
        #     posterior_samples = self.posteriorNat.prepare_posterior_samples_from_prior_weights(prior_weights)

        return self.priorNat.prior_klds, self.posteriorNat.klds, out

    @property
    def is_cuda(self):
        """
        Check if saved_models parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save saved_models with its parameters to the given path. Conventionally the
        path should end with '*.saved_models'.

        Inputs:
        - path: path string
        """
        print('Saving saved_models... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the outout after the saved_models is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.priorNat.enable_test_dropout()

        with torch.no_grad():
            self.is_training = False
            out = self.forward(X)
            out = out[2]

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction
