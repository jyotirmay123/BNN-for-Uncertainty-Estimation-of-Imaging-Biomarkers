import glob
import os

import numpy as np
import torch
from nn_common_modules import losses as additional_losses
from torch.optim import lr_scheduler

from utils.common_utils import CommonUtils
from utils.log_utils import LogWriter

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=additional_losses.KLDCECombinedLoss(),
                 model_name='hquicknat',
                 labels=None,
                 num_epochs=10,
                 log_nth=5,
                 lr_scheduler_step_size=5,
                 lr_scheduler_gamma=0.5,
                 use_last_checkpoint=True,
                 exp_dir='experiments',
                 log_dir='logs'):

        self.device = device
        self.model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs

        if torch.cuda.is_available():
            self.loss_func = loss_func.cuda(device)
            self.loss_func_kl_div = additional_losses.KLDivLossFunc().cuda(device)
        else:
            self.loss_func = loss_func
            self.loss_func_kl_div = additional_losses.KLDivLossFunc()

        self.optim = optim(self.model.parameters(), **optim_args)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size,
                                             gamma=lr_scheduler_gamma)

        exp_dir_path = os.path.join(exp_dir, exp_name)
        CommonUtils.create_if_not(exp_dir_path)
        CommonUtils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint

        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0

        if use_last_checkpoint:
            self.load_checkpoint()

        self.model_id = self.which_architecture()

        print("Solver started with model: {0} and model_id: {1}".format(self.model_name, self.model_id))

    # TODO:Need to correct the CM and dice score calculation.
    def train(self, train_loader, val_loader):
        """
        Train a given saved_models with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print('START TRAINING. : saved_models name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            for phase in ['train', 'val']:
                print("<<<= Phase: %s =>>>" % phase)

                dice_loss_arr = []
                ce_loss_arr = []
                kld_loss_arr = []
                loss_arr = []

                out_list = []
                y_list = []

                if phase == 'train':
                    model.train()
                    scheduler.step(epoch)
                else:
                    model.eval()
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    X = sample_batched[0].type(torch.FloatTensor)
                    y = sample_batched[1].type(torch.FloatTensor)
                    w = sample_batched[2].type(torch.FloatTensor)
                    cw = sample_batched[3].type(torch.FloatTensor)

                    if model.is_cuda:
                        X, y = X.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True)
                        w, cw = w.cuda(self.device, non_blocking=True), cw.cuda(self.device, non_blocking=True)

                    priors, posteriors = None, None
                    if self.model_id == 1:
                        samples_generator = model.sample_generator(X, y)
                        priors, posteriors = next(samples_generator)

                    if phase == 'train':
                        model.set_is_training(True)
                        intermediate_output = model(X, y)
                    else:
                        model.set_is_training(False)
                        intermediate_output = model(X)

                    # None to calculate normal dice score, False to ignore cross_entropy. #setting for hquicknat,
                    # quicknat needs weighted CE and punet need to figure out
                    if self.model_id == 0:  # Quicknat
                        output = intermediate_output[2]
                        intermediate_loss = self.loss_func(intermediate_output, y, (None, False))
                    elif self.model_id == 1:  # Punet
                        output = intermediate_output
                        intermediate_loss = self.loss_func((priors, posteriors, output), y, (None, False))
                    else:  # Hquicknat
                        output = intermediate_output[2]
                        intermediate_loss = self.loss_func(intermediate_output, y, (None, False))

                    dice_loss = intermediate_loss[0]
                    ce_loss = intermediate_loss[1]
                    kl_div_loss = intermediate_loss[2]
                    loss = intermediate_loss[3]

                    if phase == 'train':
                        optim.zero_grad()
                        loss = loss.cuda()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration,
                                                         loss_name='loss')
                            self.logWriter.loss_per_iter(dice_loss.item(), i_batch, current_iteration,
                                                         loss_name='dice_loss')
                            self.logWriter.loss_per_iter(ce_loss.item(), i_batch, current_iteration,
                                                         loss_name='ce_loss')
                            self.logWriter.loss_per_iter(kl_div_loss.item(), i_batch, current_iteration,
                                                         loss_name='kl_div_loss')

                            # self.logWriter.graph(model, (X, y))
                        current_iteration += 1

                    loss_arr.append(loss.item())
                    ce_loss_arr.append(ce_loss.item())
                    dice_loss_arr.append(dice_loss.item())
                    kld_loss_arr.append(kl_div_loss.item())

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())

                    del X, y, w, cw, intermediate_output, output, batch_output, loss, ce_loss, \
                        dice_loss, kl_div_loss, intermediate_loss, priors, posteriors
                    torch.cuda.empty_cache()

                    if phase == 'val':
                        if i_batch != len(dataloaders[phase]) - 1:
                            print("#", end='', flush=True)
                        else:
                            print("100%", flush=True)

                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'start_iteration': current_iteration + 1,
                    'arch': self.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

                with torch.no_grad():
                    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)
                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch, loss_name='loss')
                    self.logWriter.loss_per_epoch(ce_loss_arr, phase, epoch, loss_name='ce_loss')
                    self.logWriter.loss_per_epoch(dice_loss_arr, phase, epoch, loss_name='dice_loss')
                    self.logWriter.loss_per_epoch(kld_loss_arr, phase, epoch, loss_name='kl_div_loss')

                    index = np.random.choice(len(dataloaders[phase].dataset.X), 3, replace=False)
                    if phase == 'val':
                        self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                       model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                       dataloaders[phase].dataset.y[index],
                                                       dataloaders[phase].dataset.X[index], phase, epoch)
                    else:
                        # TODO: Dataloader image is not correct.
                        self.logWriter.image_per_epoch(out_arr[index], out_arr[index], y_arr[index],
                                                       dataloaders[phase].dataset.X[index], phase, epoch)
                    self.logWriter.cm_per_epoch(phase, out_arr, y_arr, epoch)
                    ds_mean = self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)
                    if phase == 'val':
                        if ds_mean > self.best_ds_mean:
                            self.best_ds_mean = ds_mean
                            self.best_ds_mean_epoch = epoch

            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")

        print('FINISH.')
        self.logWriter.close()

    def which_architecture(self):
        if 'HQUICKNAT' in self.model_name.upper():
            return 2
        elif 'QUICKNAT' in self.model_name.upper():
            return 0
        elif 'PUNET' in self.model_name.upper():
            return 1
        else:
            raise Exception('Cannot able to locate loss function for current {}'.format(self.model_name))

    def save_best_model(self, path):
        """
        Save saved_models with its parameters to the given path. Conventionally the
        path should end with "*.saved_models".
        Inputs:
        - path: path string
        """
        print('Saving saved_models... %s' % path)
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self, epoch=None):
        if epoch is not None:
            checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                           'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION)
            self._load_checkpoint_file(checkpoint_path)
        else:
            all_files_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
            list_of_files = glob.glob(all_files_path)
            if len(list_of_files) > 0:
                checkpoint_path = max(list_of_files, key=os.path.getctime)
                self._load_checkpoint_file(checkpoint_path)
            else:
                self.logWriter.log(
                    "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))

    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])

        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))
