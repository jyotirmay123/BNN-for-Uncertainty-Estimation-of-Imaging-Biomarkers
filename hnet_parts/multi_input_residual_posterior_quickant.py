"""Residual Quicknat architecture"""
from quicknat import QuickNat
import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se


class MultiInputResidualPosteriorQuickNat(QuickNat):
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
        super(MultiInputResidualPosteriorQuickNat, self).__init__(params)

        params['num_filters'] = 64
        params['num_channels'] = 2
        self.pos_encode1 = sm.EncoderBlock(params, se_block_type=se.SELayer.NONE)
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

        self.alternate_sample_pick = False
        self.is_training = True
        self.no_of_samples = params['latent_variables']
        self.posterior_samples = {}
        self.posteriors = {}

    def forward(self, inp, ground_truth=None):

        inp_and_gt = torch.cat([inp, ground_truth.unsqueeze(dim=1)], dim=1)
        pos_e1, pos_out1, pos_ind1 = self.pos_encode1.forward(inp_and_gt)
        pos_e2, pos_out2, pos_ind2 = self.encode2.forward(pos_e1)
        pos_e3, pos_out3, pos_ind3 = self.encode3.forward(pos_e2)
        pos_e4, pos_out4, pos_ind4 = self.encode4.forward(pos_e3)

        pos_bn = self.bottleneck.forward(pos_e4)

        pos_d4 = self.decode1.forward(pos_bn, pos_out4, pos_ind4)
        pos_1_sample = self.prior_posterior_block(pos_bn, 1, 'layer1')
        self.posterior_samples['layer1'] = pos_1_sample
        pos_concat_d4 = self.concat(pos_d4, pos_1_sample, pos_1_sample)
        pos_res_d4 = self.resBlock1.forward(pos_concat_d4, depth=2)

        pos_d3 = self.decode2.forward(pos_res_d4, pos_out3, pos_ind3)

        pos_2_sample = self.prior_posterior_block(pos_res_d4, 2, 'layer2')
        self.posterior_samples['layer2'] = pos_2_sample
        pos_concat_d3 = self.concat(pos_d3, pos_2_sample, pos_2_sample)
        pos_res_d3 = self.resBlock2.forward(pos_concat_d3, depth=2)

        pos_d2 = self.decode2.forward(pos_res_d3, pos_out2, pos_ind2)

        pos_3_sample = self.prior_posterior_block(pos_res_d3, 3, 'layer3')
        self.posterior_samples['layer3'] = pos_3_sample
        pos_concat_d2 = self.concat(pos_d2, pos_3_sample, pos_3_sample)
        pos_res_d2 = self.resBlock3.forward(pos_concat_d2, depth=2)

        pos_d1 = self.decode2.forward(pos_res_d2, pos_out1, pos_ind1)

        pos_4_sample = self.prior_posterior_block(pos_res_d2, 4, 'layer4')

        self.posterior_samples['layer4'] = pos_4_sample
        pos_concat_d1 = self.concat(pos_d1, pos_4_sample, pos_4_sample)
        pos_res_d1 = self.resBlock4.forward(pos_concat_d1, depth=2)

        classifier_input = pos_res_d1

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

    def prior_posterior_block(self, inp, depth, layer_id=None):
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

        if layer_id is not None:
            self.posteriors[layer_id] = {}
            self.posteriors[layer_id]['mu'] = mu
            self.posteriors[layer_id]['sigma'] = sigma

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

    def get_posterior_samples(self, batch_size=None):
        if not self.is_training:
            for k in self.posteriors.keys():
                samples = self.reparameterize(self.posteriors[k]['mu'], self.posteriors[k]['sigma'])
                if batch_size is None:
                    raise Exception('Batch size cannot be None while sampling in validation time from posteriors.')
                self.posterior_samples[k] = samples[:batch_size]
        return self.posterior_samples

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
