import glob
import os

import numpy as np
import torch
from ncm import losses as additional_losses
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
                 posterior_optim = torch.optim.Adam,
                 optim_args={},
                 loss_func=additional_losses.CombinedLoss(),
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
        self.model, self.posterior_model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs

        # TODO: Fit the additional objective function into the architecture properly later
        if torch.cuda.is_available():
            self.loss_func = loss_func.cuda(device)
            self.posterior_loss = additional_losses.CombinedLoss().cuda(device)
        else:
            self.loss_func = loss_func
            self.posterior_loss = additional_losses.CombinedLoss()

        self.optim = optim(self.model.parameters(), **optim_args)
        self.posterior_optim = posterior_optim(self.posterior_model.parameters(), **optim_args)
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

    # TODO:Need to correct the CM and dice score calculation.
    def train(self, train_loader, val_loader):
        """
        Train a given saved_models with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler
        posterior_model, posterior_optim = self.posterior_model, self.posterior_optim
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        warmup_posterior = True
        run_warmup = 5
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)
            posterior_model.cuda(self.device)

        print('START TRAINING. : saved_models name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            for phase in ['train', 'val']:
                print("<<<= Phase: %s =>>>" % phase)
                loss_arr = [0]
                ce_loss_arr = [0]
                dice_loss_arr = [0]
                kld_loss_arr = [0]
                posterior_loss_arr = [0]
                out_list = []
                y_list = []
                if phase == 'train':
                    model.train()
                    posterior_model.train()
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

                    if phase == 'train':
                        if warmup_posterior:
                            for i in range(run_warmup):
                                posterior_out = posterior_model(X, y)
                                posterior_loss = self.posterior_loss.forward(posterior_out, y, (w, cw))
                                posterior_optim.zero_grad()
                                posterior_loss.backward(retain_graph=True)
                                posterior_optim.step()
                                posterior_loss_arr.append(posterior_loss.item())
                                if posterior_loss.item() < 0.01:
                                    run_warmup = 1

                        posterior_model(X, y)
                        model.is_training = True
                        posterior_samples = posterior_model.get_samples()
                        model.set_posterior_samples(posterior_samples)
                        prior_samples = model.get_prior_samples()
                        output = model(X)
                    else:
                        model.is_training = False
                        output = model(X)
                        prior_weights = model.get_prior_weights_for_posterior_samplings()
                        prior_samples = model.get_prior_samples()
                        posterior_samples = posterior_model.prepare_posterior_samples_from_prior_weights(prior_weights)

                    intermediate_loss = self.loss_func((prior_samples, posterior_samples, output), y, (w, cw))

                    dice_loss = intermediate_loss[0]
                    ce_loss = intermediate_loss[1]
                    kl_div_loss = intermediate_loss[2]
                    loss = intermediate_loss[3]

                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                            self.logWriter.dice_loss_per_iter(dice_loss.item(), i_batch, current_iteration)
                            self.logWriter.ce_loss_per_iter(ce_loss.item(), i_batch, current_iteration)
                            self.logWriter.kldiv_loss_per_iter(kl_div_loss.item(), i_batch, current_iteration)
                            self.logWriter.posterior_loss_per_iter(posterior_loss_arr[-1], i_batch, current_iteration)
                            # self.logWriter.graph(model, (X, y))
                        current_iteration += 1

                    loss_arr.append(loss.item())
                    ce_loss_arr.append(ce_loss.item())
                    dice_loss_arr.append(dice_loss.item())
                    kld_loss_arr.append(kl_div_loss.item())

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())

                    del X, y, output, batch_output, loss, intermediate_loss
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
                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    self.logWriter.ce_loss_per_epoch(ce_loss_arr, phase, epoch)
                    self.logWriter.dice_loss_per_epoch(dice_loss_arr, phase, epoch)
                    self.logWriter.kldiv_loss_per_epoch(kld_loss_arr, phase, epoch)
                    self.logWriter.posterior_loss_per_epoch(posterior_loss_arr, phase, epoch)
                    index = np.random.choice(len(dataloaders[phase].dataset.X), 3, replace=False)
                    self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                   dataloaders[phase].dataset.y[index],
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
