import os
import abc
import torch
import torch.nn.init as init
import shutil

torch.set_default_tensor_type('torch.FloatTensor')


class ExecutorInterface(abc.ABC):
    def __init__(self, settings):
        super().__init__(settings)

    @staticmethod
    def delete_contents(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    @abc.abstractmethod
    def train(self, train_params, common_params, data_params, net_params, utils=None):
        pass

    @abc.abstractmethod
    def evaluate(self, eval_params, net_params, data_params, common_params, train_params, utils=None):
        pass