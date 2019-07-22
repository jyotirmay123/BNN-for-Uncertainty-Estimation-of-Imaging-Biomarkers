import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from utils.surface_distance import compute_surface_distances, compute_surface_overlap_at_tolerance
from utils.data_utils import DataUtils


class Evaluator(DataUtils):
    def __init__(self, settings):
        super().__init__(settings)

    @staticmethod
    def IOU_score(batch_outputs):
        means = np.mean(batch_outputs, axis=0)
        sigmas = np.var(batch_outputs, axis=0)
        IOUScore = np.divide(means, sigmas, out=np.zeros_like(means), where=sigmas != 0)
        return IOUScore

    @staticmethod
    def voxelwise_uncertainty(self, input):
        pass

    @staticmethod
    def dice_confusion_matrix(vol_output, ground_truth, num_class, no_samples=10, mode='train'):
        dice_cm = torch.zeros(num_class, num_class)
        if mode == 'train':
            samples = np.random.choice(len(vol_output), no_samples)
            vol_output, ground_truth = vol_output[samples], ground_truth[samples]
        for i in range(num_class):
            GT = (ground_truth == i).float()
            for j in range(num_class):
                Pred = (vol_output == j).float()
                inter = torch.sum(torch.mul(GT, Pred))
                union = torch.sum(GT) + torch.sum(Pred) + 0.0001
                dice_cm[i, j] = 2 * torch.div(inter, union)
        avg_dice = torch.mean(torch.diagflat(dice_cm))
        return avg_dice, dice_cm

    @staticmethod
    def dice_score_perclass(vol_output, ground_truth, num_class, no_samples=10, mode='train'):
        dice_perclass = torch.zeros(num_class)
        if mode == 'train':
            samples = np.random.choice(len(vol_output), no_samples)
            vol_output, ground_truth = vol_output[samples], ground_truth[samples]

        for i in range(num_class):
            GT = (ground_truth == i).float()
            Pred = (vol_output == i).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_perclass[i] = (2 * torch.div(inter, union))
        return dice_perclass

    def dice_surface_distance_perclass(self, vol_output, ground_truth, no_samples=10, mode='train'):
        surface_distance_perclass = np.zeros((self.num_class, 2))
        if mode == 'train':
            samples = np.random.choice(len(vol_output), no_samples)
            vol_output, ground_truth = vol_output[samples], ground_truth[samples]
        for i in range(self.num_class):
            GT = (ground_truth == i).int()
            Pred = (vol_output == i).int()
            surface_dis = compute_surface_distances(GT.cpu().numpy(), Pred.cpu().numpy(), self.target_voxel_dimension)
            avg_surface_distance = compute_surface_overlap_at_tolerance(surface_dis, self.organ_tolerances[i])
            surface_distance_perclass[i] = avg_surface_distance

        return surface_distance_perclass

    def evaluate_dice_score(self, prediction_path, load_from_txt_file=True, device=0, logWriter=None,
                            is_train_phase=False):
        mode = 'train' if is_train_phase else 'eval'
        print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

        batch_size = self.eval_batch_size

        with open(self.test_volumes) as file_handle:
            volumes_to_use = file_handle.read().splitlines()

        model = torch.load(self.eval_model_path)
        if self.model_name == 'quicknat':
            model.enable_test_dropout()
        else:
            model.is_training = False

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.empty_cache()
            model.cuda(device)

        model.eval()
        volume_dice_score_list = []
        volume_surface_distance_list = []
        print("Evaluating now...")
        file_paths = self.load_file_paths(load_from_txt_file=load_from_txt_file, is_train_phase=is_train_phase)
        with torch.no_grad():
            for vol_idx, file_path in enumerate(file_paths):
                print(file_path)
                volume, labelmap, header, weights, class_weights = self.load_and_preprocess(file_path)

                volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
                volume = torch.tensor(np.ascontiguousarray(volume)).type(torch.FloatTensor)
                labelmap = torch.tensor(np.ascontiguousarray(labelmap)).type(torch.FloatTensor)

                volume_prediction = []
                heat_map_arr = []
                for i in range(0, len(volume), batch_size):
                    batch_x = volume[i: i + batch_size]
                    # batch_y = labelmap[i:i + batch_size]
                    if cuda_available:
                        batch_x = batch_x.cuda(device)

                    out = model(batch_x)
                    _, batch_output = torch.max(out, dim=1)
                    if self.is_uncertainity_check_enabled:
                        batch_uncertainty_outputs = []
                        for _ in range(self.mc_sample):
                            out = model.forward(batch_x)
                            out = F.softmax(out, dim=1)
                            batch_output_ = (out.cpu().numpy()).astype('float32')
                            batch_output_ = np.squeeze(batch_output_)
                            batch_uncertainty_outputs.append(batch_output_)

                        batch_uncertainty_outputs = np.array(batch_uncertainty_outputs)
                        heat_map_over_class = -np.sum(
                            np.multiply(batch_uncertainty_outputs, np.log(batch_uncertainty_outputs + 0.0001)), axis=0)
                        heat_map = np.sum(heat_map_over_class, axis=1)
                        heat_map_output = torch.from_numpy(heat_map).float().to(device)
                        heat_map_arr.append(heat_map_output)

                        infer = np.mean(batch_uncertainty_outputs, axis=0)
                        batch_output = np.argmax(infer, axis=1)
                        batch_output = torch.from_numpy(batch_output).float().to(device)

                    volume_prediction.append(batch_output)

                self.create_if_not(prediction_path)
                if self.is_uncertainity_check_enabled:
                    heat_map_arr = torch.cat(heat_map_arr, dim=0)
                    heat_map_arr = (heat_map_arr.cpu().numpy()).astype('float32')
                    heat_map_nifti_img = nib.MGHImage(np.squeeze(heat_map_arr), np.eye(4), header=header)
                    nib.save(heat_map_nifti_img,
                             os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_uncertainty.nii.gz')))

                volume_prediction = torch.cat(volume_prediction)

                volume_prediction_ = (volume_prediction.cpu().numpy()).astype('float32')
                nifti_img = nib.MGHImage(np.squeeze(volume_prediction_), np.eye(4), header=header)
                nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_seg.nii.gz')))

                volume_dice_score = self.dice_score_perclass(volume_prediction, labelmap.cuda(device), self.num_class,
                                                             mode=mode)

                volume_dice_surface_distance = self.dice_surface_distance_perclass(volume_prediction,
                                                                                   labelmap.cuda(device),
                                                                                   mode=mode)

                if logWriter:
                    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                                              vol_idx)

                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)
                print('\n', volume_dice_score, np.mean(volume_dice_score))

                volume_surface_distance_list.append(volume_dice_surface_distance)
                print('\n', volume_dice_surface_distance, np.mean(volume_dice_surface_distance[:, 0]),
                      np.mean(volume_dice_surface_distance[:, 1]))

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.mean(dice_score_arr)
            class_dist = [dice_score_arr[:, c] for c in range(self.num_class)]

            surface_distance_arr = np.asarray(volume_surface_distance_list)

            if logWriter:
                logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
            print("DONE")

        return avg_dice_score, class_dist, dice_score_arr, surface_distance_arr
