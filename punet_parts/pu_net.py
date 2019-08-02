from .encoder_net import EncoderNet
from .multi_input_quicknat import MultiInputQuickNat
import torch.nn as nn
import torch
import numpy as np


class ProbabilisticUNet(nn.Module):
    """
    Probabilistic QuickNat architectured. Inspired from Probabilistic U-Net.
    """

    def __init__(self, params):
        super(ProbabilisticUNet, self).__init__()
        # Initialisation of multi input variant of QuickNat
        self.quickNat = MultiInputQuickNat(params)
        # Prior net forward pass will get input image only.
        params['num_channels'] = 1
        self.priorNet = EncoderNet(params)
        params['num_channels'] = 2
        # Posterior net forward pass will get 2 inp i.e. inp image and ground truth.
        self.posteriorNet = EncoderNet(params)
        self.alternate_sample_pick = True
        self.sample = None
        self.uncertainty_check = params['uncertainty_check']
        self.sampling_time = params['sampling_frequency']
        self.is_training = True

    def set_is_training(self, is_training):
        self.is_training = is_training

    def forward(self, inp, ground_truth=None):
        """

        :param input: X
        :param ground_truth: y
        :return: generaed samples from prior and posterior encoder network and predicted segmentation map.
        """
        # if not self.is_training:
        #     self.quickNat.enable_test_dropout()

        if ground_truth is not None:  # Training Time
            y_out = self.quickNat.forward(inp, self.sample)
            return y_out
        else:  # Testing Time
            return next(self.y_out_generator(inp))

    def enable_test_dropout(self):
        """
        Enables test time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    def y_out_generator(self, inp):
        # # Prior Encoder network getting trained on input image only.
        prior_mu, prior_sigma = self.priorNet.forward(inp)
        while True:
            prior_sample = self.priorNet.reparameterize(prior_mu, prior_sigma)
            # QuickNAt trained on input image and samples from posterior network.
            y_out = self.quickNat.forward(inp, prior_sample)
            yield y_out
            if not self.uncertainty_check:
                break

    def sample_generator(self, input, ground_truth):

        # Prior Encoder network getting trained on input image.
        prior_mu, prior_sigma = self.priorNet.forward(input)
        # Posterior Encoder network trained with input image conditioned on ground truth.
        posterior_mu, posterior_sigma = self.posteriorNet.forward(input, ground_truth)

        # QuickNAt trained on input image and samples from posterior network.
        for i in range(self.sampling_time):
            #prior_sample = self.priorNet.reparameterize(prior_mu, prior_sigma)
            posterior_sample = self.posteriorNet.reparameterize(posterior_mu, posterior_sigma)
            self.sample = posterior_sample
            yield {'mu_sigma': (prior_mu, prior_sigma)}, {'mu_sigma': (posterior_mu, posterior_sigma)}

    @staticmethod
    def gaussian(ins, is_training, mean, stddev):
        from torch.autograd import Variable
        if is_training:
            noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
            # print(noise)
            return ins + noise
        return ins

    @property
    def is_cuda(self):
        """
        Check if saved_models parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

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

        with torch.no_grad():
            out = self.forward(X)

        if enable_dropout:
            self.quickNat.enable_test_dropout()

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction

    def save(self, path):
        """
        Save saved_models with its parameters to the given path. Conventionally the
        path should end with '*.saved_models'.

        Inputs:
        - path: path string
        """
        print('Saving saved_models... %s' % path)
        torch.save(self, path)
