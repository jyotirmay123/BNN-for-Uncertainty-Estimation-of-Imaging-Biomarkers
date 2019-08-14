"""Multi Input Hierarchical Quicknat - Posterior Leg architecture"""
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se


class MultiInputResidualPosteriorQuickNat(nn.Module):
    """
    A PyTorch implementation of Hierarchical Quicknat - Posterior

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
        super(MultiInputResidualPosteriorQuickNat, self).__init__()

        # params['num_filters'] = 64
        params['num_channels'] = 2
        self.encode1 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode3 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode4 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.bottleneck = sm.DenseBlock(params, se_block_type=se.SELayer.CSSE)

        self.resBlock1 = sm.FullyPreActivatedResBlock(params, 1)
        self.resBlock2 = sm.FullyPreActivatedResBlock(params, 2)
        self.resBlock3 = sm.FullyPreActivatedResBlock(params, 4)
        self.resBlock4 = sm.FullyPreActivatedResBlock(params, 8)

        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode2 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode3 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode4 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)

        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

        self.fc_mu1 = nn.Linear(64, 1)
        self.fc_sigma1 = nn.Linear(64, 1)

        self.fc_mu2 = nn.Linear(64, 2)
        self.fc_sigma2 = nn.Linear(64, 2)

        self.fc_mu3 = nn.Linear(64, 4)
        self.fc_sigma3 = nn.Linear(64, 4)

        self.fc_mu4 = nn.Linear(64, 8)
        self.fc_sigma4 = nn.Linear(64, 8)

        self.samples = {}
        self.klds = {}

    def forward(self, inp, ground_truth=None):

        inp_and_gt = torch.cat([inp, ground_truth.unsqueeze(dim=1)], dim=1)
        e1, out1, ind1 = self.encode1.forward(inp_and_gt)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)

        bn = self.bottleneck.forward(e4)

        d4 = self.decode1.forward(bn, out4, ind4)
        layer1_sample, kld_1 = self.posterior_block(bn, 1)
        self.samples['layer1'] = layer1_sample
        self.klds['layer1'] = kld_1
        concat_d4 = self.concat(d4, layer1_sample)
        res_d4 = self.resBlock1.forward(concat_d4, depth=4)

        d3 = self.decode2.forward(res_d4, out3, ind3)

        layer2_sample, kld_2 = self.posterior_block(res_d4, 2)
        self.samples['layer2'] = layer2_sample
        self.klds['layer2'] = kld_2
        concat_d3 = self.concat(d3, layer2_sample)
        res_d3 = self.resBlock2.forward(concat_d3, depth=3)

        d2 = self.decode2.forward(res_d3, out2, ind2)

        layer3_sample, kld_3 = self.posterior_block(res_d3, 3)
        self.samples['layer3'] = layer3_sample
        self.klds['layer3'] = kld_3
        concat_d2 = self.concat(d2, layer3_sample)
        res_d2 = self.resBlock3.forward(concat_d2, depth=2)

        d1 = self.decode2.forward(res_d2, out1, ind1)

        layer4_sample, kld_4 = self.posterior_block(res_d2, 4)
        self.samples['layer4'] = layer4_sample
        self.klds['layer4'] = kld_4
        concat_d1 = self.concat(d1, layer4_sample)
        res_d1 = self.resBlock4.forward(concat_d1, depth=1)

        # return self.samples
        return self.classifier(res_d1)

    def concat(self, input_tensor, sample):
        sample = sample.unsqueeze(dim=2).unsqueeze(dim=2)
        expanded_inp = sample.expand(-1, sample.size(1), input_tensor.size(2), input_tensor.size(3))
        concated_tensor = torch.cat((input_tensor, expanded_inp), 1)
        return concated_tensor

    def prepare_posterior_samples_from_prior_weights(self, prior_weights):
        samples = {}
        for i, k in enumerate(prior_weights):
            weights = prior_weights[k]
            samples[k] = self.posterior_block(weights, i + 1)
        return samples

    def posterior_block(self, inp, depth):
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
        kld = self.calculate_kld(mu, sigma)
        sample = self.reparameterize(mu, sigma)
        return sample, kld

    def calculate_kld(self, mu, logvar):
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (mu, logvar)

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

    def reparameterize(self, mu, logvar):
        """

        :param mu: mean
        :param logvar: variance
        :return: randomly generated sample
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_samples(self):
        return self.samples
