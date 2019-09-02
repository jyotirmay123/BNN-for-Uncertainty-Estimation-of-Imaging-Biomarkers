import os
import numpy as np
import torch

from interfaces.solver_interface import SolverInterface

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(SolverInterface):

    def __init__(self, **args):
        super().__init__(**args)

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
            'val': val_loader,
            'train': train_loader
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
                    # w = sample_batched[2].type(torch.FloatTensor)
                    # cw = sample_batched[3].type(torch.FloatTensor)

                    if model.is_cuda:
                        X, y = X.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True)
                        # w, cw = w.cuda(self.device, non_blocking=True), cw.cuda(self.device, non_blocking=True)

                    if phase == 'train':
                        model.set_is_training(True)
                    else:
                        model.set_is_training(False)

                    intermediate_output = model.forward(X)
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
                            self.notifier.notify(
                                f'{self.model_name.upper()} NOTIFICATION EPOCH {epoch}, ITERATION: {current_iteration}'
                                f' :: dice_loss: {dice_loss}, ce_loss: {ce_loss}, kl_loss: {kl_div_loss}, loss: {loss}'
                            )
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

                    del X, y, intermediate_output, output, batch_output, loss, ce_loss, \
                        dice_loss, kl_div_loss, intermediate_loss
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
                        # model.predict(dataloaders[phase].dataset.X[index], self.device)
                        self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                       model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                       [dataloaders[phase].dataset.y[index]],
                                                       dataloaders[phase].dataset.X[index], phase, epoch)
                    else:
                        # TODO: Dataloader image is not correct.
                        self.logWriter.image_per_epoch(out_arr[index], out_arr[index], [y_arr[index]],
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
