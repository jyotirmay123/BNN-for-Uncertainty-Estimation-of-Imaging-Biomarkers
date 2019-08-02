import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
# Does IOU for a structure.
from medpy.metric import jc
import matplotlib.pyplot as plt
from utils.common_utils import CommonUtils
from utils.surface_distance import compute_surface_distances, compute_surface_overlap_at_tolerance
from utils.data_utils import DataUtils


class Evaluator(DataUtils):
    def __init__(self, settings):
        super().__init__(settings)

    def ncc(self, a, v, zero_norm=True):

        a = a.flatten()
        v = v.flatten()

        if zero_norm:

            a = (a - np.mean(a)) / (np.std(a) * len(a))
            v = (v - np.mean(v)) / np.std(v)

        else:

            a = (a) / (np.std(a) * len(a))
            v = (v) / np.std(v)

        return np.correlate(a, v)

    def variance_ncc_dist(self, sample_arr, gt_arr):
        print(sample_arr.shape, gt_arr.shape)

        def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

            log_samples = np.log(m_samp + eps)

            return -1.0 * np.sum(m_gt * log_samples, axis=-1)

        """
        :param sample_arr: expected shape N x X x Y 
        :param gt_arr: M x X x Y
        :return: 
        """

        mean_seg = np.mean(sample_arr, axis=0)

        N = sample_arr.shape[0]
        M = gt_arr.shape[0]

        sX = sample_arr.shape[1]
        sY = sample_arr.shape[2]

        E_ss_arr = np.zeros((N, sX, sY))
        for i in range(N):
            E_ss_arr[i, ...] = pixel_wise_xent(sample_arr[i, ...], mean_seg)
            #plt.imsave(f'/home/abhijit/Jyotirmay/thesis/hquicknat/snapshots/ess/ess_{i}.png', E_ss_arr[i,...], cmap='Greys')

        E_ss = np.mean(E_ss_arr, axis=0)

        E_sy_arr = np.zeros((M, N, sX, sY))
        for j in range(M):
            for i in range(N):
                E_sy_arr[j, i, ...] = pixel_wise_xent(sample_arr[i, ...], gt_arr[j, ...])
                #plt.imsave(f'/home/abhijit/Jyotirmay/thesis/hquicknat/snapshots/ess/esy_{i}.png', np.squeeze(E_sy_arr[0, i, ...]), cmap='Greys')

        E_sy = np.mean(E_sy_arr, axis=1)

        ncc_list = []
        for j in range(M):
            ncc_list.append(self.ncc(E_ss, E_sy[j, ...]))

        return np.abs((1 / M) * sum(ncc_list))

    def generalised_energy_distance(self, sample_arr, gt_arr, nlabels, **kwargs):

        def dist_fct(m1, m2):

            label_range = kwargs.get('label_range', range(nlabels))

            per_label_iou = []
            for lbl in label_range:
                if lbl == 0:
                    continue
                # assert not lbl == 0  # tmp check
                m1_bin = (m1 == lbl) * 1
                m2_bin = (m2 == lbl) * 1

                if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                    per_label_iou.append(1)
                elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                    per_label_iou.append(0)
                else:
                    per_label_iou.append(jc(m1_bin, m2_bin))

            # print(1-(sum(per_label_iou) / nlabels))

            return 1 - (sum(per_label_iou) / nlabels)

        """
        :param sample_arr: expected shape N x X x Y 
        :param gt_arr: M x X x Y
        :return: 
        """

        N = sample_arr.shape[0]
        M = gt_arr.shape[0]

        d_sy = []
        d_ss = []
        d_yy = []

        for i in range(N):
            for j in range(M):
                # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
                d_sy.append(dist_fct(sample_arr[i, ...], gt_arr[j, ...]))

        for i in range(N):
            for j in range(N):
                # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
                d_ss.append(dist_fct(sample_arr[i, ...], sample_arr[j, ...]))

        for i in range(M):
            for j in range(M):
                # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
                d_yy.append(dist_fct(gt_arr[i, ...], gt_arr[j, ...]))

        return (2. / (N * M)) * sum(d_sy) - (1. / N ** 2) * sum(d_ss) - (1. / M ** 2) * sum(d_yy)

    def intersection_overlap_per_structure_per_slice(self, samples_):

        iou_per_slice = [0, 0]
        for i in range(samples_.shape[1]):
            samples = samples_[:, i]
            for c in range(self.num_class):
                if c == 0:
                    continue
                inter = (samples[0] == c).astype('int')
                union = (samples[0] == c).astype('int')
                for s in range(1, self.mc_sample):
                    nxt = (samples[s] == c).astype('int')
                    inter = np.multiply(inter, nxt)
                    union = (np.add(union, nxt) > 0).astype('int')
                s_inter, s_union = np.sum(inter), np.sum(union) + 0.0001
                p_diff = (s_union - s_inter)/s_union
                #print(f'slice {i} for class {c} with intersection {s_inter} and union {s_union} with % diff {p_diff}')
                iou_per_slice[c-1] += np.divide(s_inter, s_union)
        iou_s = np.divide(iou_per_slice, samples_.shape[1])
        print(iou_s)
        return iou_s

    def intersection_overlap_per_structure(self, samples):
        iou_s = []
        for c in range(self.num_class):
            if c == 0:
                continue
            inter = (samples[0] == c).astype('int')
            union = (samples[0] == c).astype('int')
            for s in range(1, self.mc_sample):
                nxt = (samples[s] == c).astype('int')
                inter = np.multiply(inter, nxt)
                union = (np.add(union, nxt) > 0).astype('int')
            s_inter, s_union = np.sum(inter), np.sum(union)
            if s_inter == 0 and s_union == 0:
                iou_s.append(1)
            elif s_inter > 0 and s_union == 0 or s_inter == 0 and s_union > 0:
                iou_s.append(0)
            iou_s.append(np.divide(s_inter, s_union))
        return iou_s

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
        return dice_perclass.numpy()

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

    def print_report(self, strin, final=False):
        if final:
            report_file = os.path.join(self.base_dir, 'reports', self.model_name, self.dataset, str(self.mc_sample)+'_' + self.ctime + '_final_report.txt')
        else:
            report_file = os.path.join(self.base_dir, 'reports', self.model_name, self.dataset, str(self.mc_sample)+'_' + self.ctime + '_report.txt')
        CommonUtils.create_file_if_not(report_file)
        report_file_obj = open(report_file, 'a+')
        print(strin, end="\n", file=report_file_obj)
        report_file_obj.close()
        print(strin)
        self.notifier.whatsapp_notifier(strin)

    def evaluate_dice_score(self, prediction_path, load_from_txt_file=True, device=0, logWriter=None,
                            is_train_phase=False):
        mode = 'train' if is_train_phase else 'eval'
        print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
        self.print_report(
            '########### Execution Settings for {0} dataset on {1} model ############## \n'.format(self.dataset, self.model_name),
            final=True)
        self.print_report(self.settings.settings_dict, final=True)
        self.print_report(self.mc_sample, final=True)
        batch_size = self.eval_batch_size

        with open(self.test_volumes) as file_handle:
            volumes_to_use = file_handle.read().splitlines()

        model = torch.load(self.eval_model_path)

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.empty_cache()
            model.cuda(device)

        model.eval()
        volume_dice_score_list = []
        volume_surface_distance_list = []
        iou_score_per_structure_list = []
        s_ncc_list = []
        s_ged_list = []

        print("Evaluating now...")
        file_paths = self.load_file_paths(load_from_txt_file=load_from_txt_file, is_train_phase=is_train_phase)
        self.print_report(
            '########### Evaluating {0} dataset on {1} model ############## \n'.format(self.dataset, self.model_name))
        self.print_report(
            '########### Evaluating {0} dataset on {1} model ############## \n'.format(self.dataset, self.model_name),
            final=True)
        with torch.no_grad():
            for vol_idx, file_path in enumerate(file_paths):
                print(file_path)
                self.print_report('# VOLUME:: ' + file_path[0].split('/')[-1].split('.')[0] + '\n')
                volume, labelmap, header, weights, class_weights = self.load_and_preprocess(file_path)

                volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
                volume = torch.tensor(np.ascontiguousarray(volume)).type(torch.FloatTensor)
                labelmap = torch.tensor(np.ascontiguousarray(labelmap)).type(torch.FloatTensor)

                volume_prediction = []
                heat_map_arr = []
                # batch_size = len(volume)
                iou_uncertainty = np.zeros((self.mc_sample, volume.shape[0], volume.shape[2], volume.shape[3]))
                for i in range(0, len(volume), batch_size):
                    batch_x = volume[i: i + batch_size]
                    # batch_y = labelmap[i:i + batch_size]
                    if cuda_available:
                        batch_x = batch_x.cuda(device)

                    out = model(batch_x)
                    # TODO: Need to integrate below logic for punet_here as well
                    if 'PUNET' not in self.model_name.upper():
                        out = out[2]
                    _, batch_output = torch.max(out, dim=1)
                    if self.is_uncertainity_check_enabled:
                        batch_uncertainty_outputs = []
                        for mcs in range(self.mc_sample):
                            model.is_training = False
                            out = model.forward(batch_x)
                            # TODO: Need to integrate below logic for punet
                            if 'PUNET' not in self.model_name.upper():
                                out = out[2]
                            out = F.softmax(out, dim=1)
                            _, batch_class_map = torch.max(out, dim=1)
                            iou_uncertainty[mcs, i: i + batch_size] = batch_class_map.cpu().numpy()
                            batch_output_ = (out.cpu().numpy()).astype('float32')
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

                s_ncc_list.append(self.variance_ncc_dist(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy()))
                s_ged_list.append(self.generalised_energy_distance(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy(), 3))

                # Save sample predictions
                for i in range(iou_uncertainty.shape[0]):
                    sample = iou_uncertainty[i]
                    heat_map_nifti_img = nib.MGHImage(np.squeeze(sample), np.eye(4), header=header)
                    sample_path = os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_samples'))
                    CommonUtils.create_if_not(sample_path)
                    nib.save(heat_map_nifti_img,
                             os.path.join(sample_path,
                                          volumes_to_use[vol_idx] + str('_sample_{}.nii.gz'.format(i))))

                iou_s = self.intersection_overlap_per_structure(iou_uncertainty)

                self.print_report('# Intersection over overlap scores per structure per volume.')
                self.print_report(f'Spleen: {iou_s[0]}   |    Liver: {iou_s[1]}  |  <-Mean: {np.mean(iou_s)} \n')

                iou_score_per_structure_list.append(iou_s)
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
                self.print_report('# Dice Score per structure per volume.')
                self.print_report(f'Spleen: {volume_dice_score[1]}   |    Liver: {volume_dice_score[2]}  |  <-Mean: {np.mean(volume_dice_score[1:])} \n')

                volume_dice_surface_distance = self.dice_surface_distance_perclass(volume_prediction,
                                                                                   labelmap.cuda(device),
                                                                                   mode=mode)

                self.print_report('# Surface distance scores per structure per volume.')

                self.print_report(f'GT TO PRED SCORES:: Spleen: {volume_dice_surface_distance[1, 0]}   |   '
                                  f'Liver: {volume_dice_surface_distance[2, 0]}  |  <-Mean: {np.mean(volume_dice_surface_distance[1:, 0])}')
                self.print_report(f'PRED TO GT SCORES::  Spleen: {volume_dice_surface_distance[1, 1]}   |   '
                                  f'Liver: {volume_dice_surface_distance[2, 1]}  |  <-Mean: {np.mean(volume_dice_surface_distance[1:, 1])}')
                self.print_report(f'  MEAN TO ABOVE  :: Spleen: {np.mean(volume_dice_surface_distance[1, :])}   |   '
                                  f'Liver: {np.mean(volume_dice_surface_distance[2, :])}  |  MEAN: {np.mean(volume_dice_surface_distance[1:])} \n')

                self.print_report(f'# Normalised Cross Correlation score: \n {s_ncc_list[-1][0]}')
                self.print_report(f'# Generalised Entropy Distance score: \n {s_ged_list[-1]}')

                if logWriter:
                    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                                              vol_idx)

                volume_dice_score = volume_dice_score
                volume_dice_score_list.append(volume_dice_score)

                # print('\n', volume_dice_score, np.mean(volume_dice_score))
                #
                volume_surface_distance_list.append(volume_dice_surface_distance)
                # print('\n', volume_dice_surface_distance, np.mean(volume_dice_surface_distance[:, 0]),
                #       np.mean(volume_dice_surface_distance[:, 1]))
                self.print_report('------------------------------------------------------------------------------ \n\n')

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.mean(dice_score_arr)
            class_dist = [dice_score_arr[:, c] for c in range(self.num_class)]

            surface_distance_arr = np.asarray(volume_surface_distance_list)
            iou_score_per_structure_arr = np.asarray(iou_score_per_structure_list)
            s_ncc_arr = np.asarray(s_ncc_list)
            s_ged_arr = np.asarray(s_ged_list)

            if logWriter:
                logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
            print("DONE")

            self.print_report('++++++++++++++++++++++++++++ FINAL REPORT +++++++++++++++++++++++++++++++\n', final=True)
            self.print_report('# Mean intersection over overlap scores.', final=True)
            self.print_report(f' OVERALL MEAN: {np.mean(iou_score_per_structure_arr)} | '
                              f'SPLEEN: {np.mean(iou_score_per_structure_arr[:, 0])} | '
                              f'LIVER: {np.mean(iou_score_per_structure_arr[:, 1])}',
                              final=True)

            self.print_report('# Mean Dice Score.', final=True)
            self.print_report(f' SPLEEN: {np.mean(dice_score_arr[:, 1])} | '
                              f'LIVER: {np.mean(dice_score_arr[:, 2])} | '
                              f'OVERALL MEAN: {np.mean(dice_score_arr[:, 1:])}\n', final=True)

            self.print_report('# Mean Surface distance scores.', final=True)
            self.print_report(f' GT TO PRED SCORES:: SPLEEN: {np.mean(surface_distance_arr[:, 1, 0])} | '
                              f'LIVER: {np.mean(surface_distance_arr[:, 2, 0])} | OVERALL MEAN: {np.mean(surface_distance_arr[:, 1:, 0])}', final=True)
            self.print_report(f' PRED TO GT SCORES:: SPLEEN: {np.mean(surface_distance_arr[:, 1, 1])} | '
                              f'LIVER: {np.mean(surface_distance_arr[:, 2, 1])} | OVERALL MEAN: {np.mean(surface_distance_arr[:, 1:, 1])}', final=True)
            self.print_report(f' OVERALL MEAN:: SPLEEN: {np.mean(surface_distance_arr[:, 1])} | '
                              f'LIVER: {np.mean(surface_distance_arr[:, 2])} | MEAN: {np.mean(surface_distance_arr[:, 1:])} \n', final=True)

            self.print_report(f'# Mean Normalised Cross Correlation score:\n {np.mean(s_ncc_arr)}', final=True)
            self.print_report(f'# Mean Generalised Entropy Distance score:\n {np.mean(s_ged_arr)}\n', final=True)

            self.print_report('-------------------------------------------------------------------------\n', final=True)

        self.print_report(
            f'############### {self.model_name} on {self.dataset} report completed ################### \n')

        return avg_dice_score, class_dist, dice_score_arr, surface_distance_arr
