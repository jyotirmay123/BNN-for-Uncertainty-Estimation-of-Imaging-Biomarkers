"""
Contains commong functions useful throughout the application
"""
import os
import torch


class CommonUtils(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_if_not(path):
        """
        Creates a folder at the given path if one doesnt exist before
        ===

        :param path: destination to check for existense
        :return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def reparameterize(mu, logvar):
        """

        :param mu: mean
        :param logvar: variance
        :return: randomly generated sample
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
