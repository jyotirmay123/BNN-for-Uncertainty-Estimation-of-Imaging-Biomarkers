import argparse
import os

import torch

from utils.evaluator import Evaluator
from hnet_parts.hquicknat import HQuicknat
from quicknat import QuickNat
from settings import compile_config
from solver import Solver
import torch.nn as nn
from utils.log_utils import LogWriter
import numpy as np
import shutil
from ncm import losses as additional_losses

torch.set_default_tensor_type('torch.FloatTensor')


class Executor(Evaluator):
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

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0.01)

    def train(self, train_params, common_params, data_params, net_params):
        global settings
        print("Loading dataset")
        train_data, test_data = self.get_imdb_dataset()
        print("Train size: %i" % len(train_data))
        print("Test size: %i" % len(test_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'],
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=train_params['val_batch_size'], shuffle=False,
                                                 num_workers=4, pin_memory=True)

        if train_params['use_pre_trained']:
            model = torch.load(train_params['pre_trained_path'])
        else:
            model = HQuicknat(net_params)
            model.apply(self.init_weights)

        # if 'hquicknat' in common_params.model_name:
        #     objective_func = additional_losses.KLDCECombinedLoss(net_params['gamma_value'],
        #                                                          net_params['beta_value'])
        # elif 'quicknat' in common_params.model_name:
        #     objective_func = additional_losses.CombinedLoss()
        # elif 'punet' in common_params.model_name:
        #     objective_func = additional_losses.KLDCECombinedLoss(net_params['gamma_value'],
        #                                                          net_params['beta_value'])
        # else:
        #     raise Exception('Cannot able to locate loss function for current {}'.format(common_params.model_name))

        solver = Solver(model,
                        device=common_params['device'],
                        num_class=net_params['num_class'],
                        optim_args={"lr": train_params['learning_rate'],
                                    "betas": train_params['optim_betas'],
                                    "eps": train_params['optim_eps'],
                                    "weight_decay": train_params['optim_weight_decay']},
                        loss_func=additional_losses.KLDCECombinedLoss(net_params['gamma_value'],
                                                                      net_params['beta_value']),
                        model_name=common_params['model_name'],
                        exp_name=train_params['exp_name'],
                        labels=data_params['labels'],
                        log_nth=train_params['log_nth'],
                        num_epochs=train_params['num_epochs'],
                        lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                        lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                        use_last_checkpoint=train_params['use_last_checkpoint'],
                        log_dir=common_params['log_dir'],
                        exp_dir=common_params['exp_dir'])

        solver.train(train_loader, val_loader)
        final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])

        model.save(final_model_path)

        print("final models saved @ " + str(final_model_path))

    def evaluate(self, eval_params, net_params, data_params, common_params, train_params):

        num_classes = net_params.num_class
        labels = data_params.labels
        device = common_params.device
        log_dir = common_params.log_dir
        exp_dir = common_params.exp_dir
        exp_name = train_params.exp_name
        save_predictions_dir = eval_params.save_predictions_dir

        prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)

        logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

        avg_dice_score, class_dist, dice_score_arr, surface_distance_arr = \
            self.evaluate_dice_score(prediction_path, True, device, logWriter, False)

        for i in range(num_classes):
            dice_score_class_mean = np.mean(dice_score_arr[:, i])
            distance_gt_to_pred_class_mean = np.mean(surface_distance_arr[:, i, 0])
            distance_pred_to_gt_class_mean = np.mean(surface_distance_arr[:, i, 1])
            print('\n###' + labels[i] + ' predictions over all Volumes: ')
            print("\n    mean dice score: " + str(dice_score_class_mean))
            print("\n    mean relative overlap on ground truth: " + str(distance_gt_to_pred_class_mean))
            print("\n    mean relative overlap on prediction: " + str(distance_pred_to_gt_class_mean))

        distance_gt_to_pred_mean = np.mean(surface_distance_arr[:, :, 0])
        distance_pred_to_gt_mean = np.mean(surface_distance_arr[:, :, 1])
        print('\n \n final mean dice score: ' + str(avg_dice_score))
        print('\n final mean relative overlap on ground truth: ' + str(distance_gt_to_pred_mean))
        print("\n final mean relative overlap on prediction: " + str(distance_pred_to_gt_mean))

        logWriter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--settings_file_path', '-cfg', required=True, help='Path to project config file(settings.ini)')
    args = parser.parse_args()

    settings = compile_config(args.settings_file_path)
    executor = Executor(settings)

    common_params = executor.common_params
    data_params = executor.data_params
    net_params = executor.net_params
    train_params = executor.train_params
    eval_params = executor.eval_params
    data_config_params = executor.data_config_params
    eval_params.update(data_config_params)

    if args.mode == 'train':
        executor.train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        executor.evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'clear':
        shutil.rmtree(os.path.join(common_params.exp_dir, train_params.exp_name))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params.log_dir, train_params.exp_name))
        print("Cleared current log directory successfully!!")

    elif args.mode == 'clear-all':
        executor.delete_contents(common_params.exp_dir)
        print("Cleared experiments directory successfully!!")
        executor.delete_contents(common_params.log_dir)
        print("Cleared logs directory successfully!!")
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')
