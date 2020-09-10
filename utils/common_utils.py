import os
import torch
from utils.notifier import Notifier
import time
import pandas as pd
from importlib.util import find_spec


class CommonUtils(object):
    def __init__(self):
        super().__init__()
        self.notifier_obj = Notifier()
        self.is_mail_notifier = None

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
    def strinfify_for_setting(str_val):
        return '"' + str(str_val) + '"'

    @staticmethod
    def create_file_if_not(path):
        """
        Creates a folder at the given path if one doesnt exist before
        ===

        :param path: destination to check for existense
        :return: None
        """
        dir = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
            file = open(path, 'w')
            file.close()

    @staticmethod
    def create_csv_file_if_not(path, cols=None):
        is_file_created_before = os.path.exists(path)
        CommonUtils.create_file_if_not(path)

        if not is_file_created_before and cols is not None:
            data = [0 for _ in range(len(cols))]
            df = pd.DataFrame([data], columns=cols, index=None)
            df.to_csv(path, columns=cols, index=None)

    @staticmethod
    def current_milli_time():
        return lambda: int(round(time.time() * 1000))

    @staticmethod
    def import_module(file, package):
        spec = find_spec(file, package=package)
        m = spec.loader.load_module()
        return m

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

    def setup_notifier(self, is_mail_notifier=True):
        self.is_mail_notifier = is_mail_notifier
        if is_mail_notifier:
            self.notifier_obj.setup_mail_notifier()
        else:
            self.notifier_obj.setup_whatsapp_notifier()

    def notify(self, message):
        if not self.notify:
            return

        if self.is_mail_notifier is None:
            raise Exception('No notifier ready to notify! Please set it up.')
        elif self.is_mail_notifier:
            self.notifier_obj.mail_notifier(message)
        else:
            self.notifier_obj.whatsapp_notifier(message)

    # TODO: Check this method functionslity.
    def launchTensorBoard(self, my_directory):
        # from tensorboard import default
        # from tensorboard import program
        # import logging
        # logging.getLogger('tensorflow').setLevel(logging.ERROR)
        # tb = program.TensorBoard(default.get_plugins())
        # tb.configure(argv=[None, '--logdir='+ my_directory])
        # tb.main()
        import subprocess
        subprocess.call(['pwd'])
        subprocess.call([f"tensorboard --logdir={my_directory}"])
        #tensorboard - -logdir = bayesian:projects / full_bayesian / logs, hierarchical: projects / hierarchical_quicknat / logs, mc_droput: projects / MC_dropout_quicknat / logs, probabilistic: projects / probabilistic_quicknat / logs

        return




