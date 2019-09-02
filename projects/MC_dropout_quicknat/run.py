import os
import torch

from interfaces.run_interface import ExecutorInterface
from utils.log_utils import LogWriter

from projects.MC_dropout_quicknat.evaluator import Evaluator
from projects.MC_dropout_quicknat.parts.quicknat import QuickNat
from projects.MC_dropout_quicknat.solver import Solver

from nn_common_modules import losses as additional_losses

torch.set_default_tensor_type('torch.FloatTensor')


class Executor(ExecutorInterface, Evaluator):
    def __init__(self, settings):
        #ExecutorInterface.__init__(self, settings)
        #Evaluator.__init__(self, settings)
        super().__init__(settings)

    def train(self, train_params, common_params, data_params, net_params, utils=None):
        print("Loading dataset")
        train_data, test_data = self.dataUtils.get_imdb_dataset()
        print("Train size: %i" % len(train_data))
        print("Test size: %i" % len(test_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'],
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=train_params['val_batch_size'], shuffle=False,
                                                 num_workers=4, pin_memory=True)

        if train_params['use_pre_trained']:
            print(f'Loading pre_trained model from: {train_params["pre_trained_path"]}')
            model = torch.load(train_params['pre_trained_path'])
        else:
            model = QuickNat(net_params)
            # model.apply(self.weights_init_orthogonal)

        solver = Solver(model=model,
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

    def evaluate(self, eval_params, net_params, data_params, common_params, train_params, utils=None):

        num_classes = net_params.num_class
        labels = data_params.labels
        device = common_params.device
        log_dir = common_params.log_dir
        exp_dir = common_params.exp_dir
        our_dir = common_params.base_dir
        exp_name = train_params.exp_name
        save_predictions_dir = eval_params.save_predictions_dir

        prediction_path = os.path.join(our_dir, 'outs', exp_name, save_predictions_dir)

        logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

        self.evaluate_dice_score(prediction_path, True, device, logWriter, False)

        logWriter.close()

