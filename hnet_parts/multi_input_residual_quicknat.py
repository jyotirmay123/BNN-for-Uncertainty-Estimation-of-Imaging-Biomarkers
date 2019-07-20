"""Residual Quicknat architecture"""
from quicknat import QuickNat
import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se


class MultiInputResidualQuickNat(QuickNat):
    """
    A PyTorch implementation of QuickNAT

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
        super(MultiInputResidualQuickNat, self).__init__(params)
        params['num_channels'] = 1
        params['num_filters'] = 64
        self.pr_encode1 = sm.EncoderBlock(params, se_block_type=se.SELayer.NONE)

        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params, se_block_type=se.SELayer.NONE)
        self.encode3 = sm.EncoderBlock(params, se_block_type=se.SELayer.NONE)
        self.encode4 = sm.EncoderBlock(params, se_block_type=se.SELayer.NONE)
        self.bottleneck = sm.DenseBlock(params, se_block_type=se.SELayer.NONE)

        self.resBlock1 = sm.FullyPreActivatedResBlock(params, 2)
        self.resBlock2 = sm.FullyPreActivatedResBlock(params, 3)
        self.resBlock3 = sm.FullyPreActivatedResBlock(params, 4)
        self.resBlock4 = sm.FullyPreActivatedResBlock(params, 5)

        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        self.decode2 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        self.decode3 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        self.decode4 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)

        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

        self.fc_mu1 = nn.Linear(64, 2)
        self.fc_sigma1 = nn.Linear(64, 2)

        self.fc_mu2 = nn.Linear(64, 3)
        self.fc_sigma2 = nn.Linear(64, 3)

        self.fc_mu3 = nn.Linear(64, 4)
        self.fc_sigma3 = nn.Linear(64, 4)

        self.fc_mu4 = nn.Linear(64, 5)
        self.fc_sigma4 = nn.Linear(64, 5)

        self.alternate_sample_pick = True
        self.is_training = True
        self.no_of_samples = params['latent_variables']
        self.posterior_samples = {}
        self.prior_samples = {}
        self.posteriors = {}

        self.prior_weights_for_posterior_samplings = {}

    def forward(self, inp, ground_truth=None):

        pr_e1, pr_out1, pr_ind1 = self.pr_encode1.forward(inp)
        pr_e2, pr_out2, pr_ind2 = self.encode2.forward(pr_e1)
        pr_e3, pr_out3, pr_ind3 = self.encode3.forward(pr_e2)
        pr_e4, pr_out4, pr_ind4 = self.encode4.forward(pr_e3)

        pr_bn = self.bottleneck.forward(pr_e4)

        pr_d4 = self.decode1.forward(pr_bn, pr_out4, pr_ind4)
        # self.prior_weights_for_posterior_samplings['layer1'] = pr_bn
        # pr_1_sample = self.prior_posterior_block(pr_bn, 1)
        # self.prior_samples['layer1'] = pr_1_sample
        # pr_concat_d4 = self.concat(pr_d4, pr_1_sample, self.posterior_samples['layer1'])
        # pr_res_d4 = self.resBlock1.forward(pr_concat_d4, depth=2)
        #
        pr_d3 = self.decode2.forward(pr_d4, pr_out3, pr_ind3)
        #
        # self.prior_weights_for_posterior_samplings['layer2'] = pr_res_d4
        # pr_2_sample = self.prior_posterior_block(pr_res_d4, 2)
        # self.prior_samples['layer2'] = pr_2_sample
        # pr_concat_d3 = self.concat(pr_d3, pr_2_sample, self.posterior_samples['layer2'])
        # pr_res_d3 = self.resBlock2.forward(pr_concat_d3, depth=2)
        #
        pr_d2 = self.decode2.forward(pr_d3, pr_out2, pr_ind2)
        #
        # self.prior_weights_for_posterior_samplings['layer3'] = pr_res_d3
        # pr_3_sample = self.prior_posterior_block(pr_res_d3, 3)
        # self.prior_samples['layer3'] = pr_3_sample
        # pr_concat_d2 = self.concat(pr_d2, pr_3_sample, self.posterior_samples['layer3'])
        # pr_res_d2 = self.resBlock3.forward(pr_concat_d2, depth=2)
        #
        pr_d1 = self.decode2.forward(pr_d2, pr_out1, pr_ind1)
        #
        # self.prior_weights_for_posterior_samplings['layer4'] = pr_res_d2
        # pr_4_sample = self.prior_posterior_block(pr_res_d2, 4)
        #
        # self.prior_samples['layer4'] = pr_4_sample
        # pr_concat_d1 = self.concat(pr_d1, pr_4_sample, self.posterior_samples['layer4'])
        # pr_res_d1 = self.resBlock4.forward(pr_concat_d1, depth=2)
        #
        classifier_input = pr_d1
        #
        # self.alternate_sample_pick = not self.alternate_sample_pick

        prob = self.classifier.forward(classifier_input)

        return prob

    def concat(self, input_tensor, prior_sample, posterior_sample=None):
        sample = posterior_sample if self.is_training and self.alternate_sample_pick else prior_sample
        sample = sample.unsqueeze(dim=2).unsqueeze(dim=2)
        expanded_inp = sample.expand(-1, sample.size(1), input_tensor.size(2), input_tensor.size(3))

        concated_tensor = torch.cat((input_tensor, expanded_inp), 1)
        return concated_tensor

    def prepare_samples_from_prior_weights(self, prior_weights):
        samples = {}
        for i, k in enumerate(prior_weights):
            weights = prior_weights[k]
            samples[k] = self.prior_posterior_block(weights, i+1)
        return samples

    def prior_posterior_block(self, inp, depth):
        inp = torch.mean(inp.view(inp.size(0), inp.size(1), -1), dim=2)
        if depth == 1:
            mu = self.fc_mu1(inp)
            sigma = self.fc_sigma1(inp)
        elif depth == 2:
            mu = self.fc_mu2(inp)
            sigma = self.fc_sigma2(inp)
        elif depth == 3:
            mu = self.fc_mu3(inp)
            sigma = self.fc_sigma3(inp)
        elif depth == 4:
            mu = self.fc_mu4(inp)
            sigma = self.fc_sigma4(inp)
        else:
            raise Exception(f'Depth cannnot be {depth}')

        sample = self.reparameterize(mu, sigma)
        return sample

    def reparameterize(self, mu, logvar):
        """

        :param mu: mean
        :param logvar: variance
        :return: randomly generated sample
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_prior_samples(self):
        return self.prior_samples

    def set_prior_samples(self, prior_samples):
        self.prior_samples = prior_samples

    def set_posterior_samples(self, posterior_samples):
        self.posterior_samples = posterior_samples

    def set_alternate_sample_pick(self, is_alternate_sample_pick_enabled):
        self.alternate_sample_pick = is_alternate_sample_pick_enabled

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
            self.enable_test_dropout()

        with torch.no_grad():
            # self.set_alternate_sample_pick(True)  # Pick samples always from trained prior block while predicting.
            self.is_training = False
            out = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction
