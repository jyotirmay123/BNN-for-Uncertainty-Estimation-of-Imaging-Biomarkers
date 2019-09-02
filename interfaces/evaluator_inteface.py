import os
import numpy as np
import torch
import nibabel as nb
from medpy.metric import jc
import abc
import pandas as pd

from utils.common_utils import CommonUtils
from utils.surface_distance import compute_surface_distances, compute_surface_overlap_at_tolerance


class EvaluatorInterface(abc.ABC):
    def __init__(self, settings):
        super().__init__()
        try:
            m = CommonUtils.import_module('.data_utils',
                                          f'dataset_groups.{settings.settings_dict["COMMON"]["dataset_groups"]}')
            self.dataUtils = m.DataUtils(settings)
        except AttributeError as ae:
            print(ae)
            import utils.extract_settings as m
            self.dataUtils = m.ExtractSettings(settings)

        self.notifier = CommonUtils()
        self.notifier.setup_notifier()

        self.csv_cols = ['project', 'dataset', 'model_name', 'volume_id', 'samples', 'sncc', 'ged', 'iou_spleen',
                         'iou_liver', 'iou_mean', 'dice_spleen', 'dice_liver', 'dice_mean',
                         'surface_distance_prtogt_spleen', 'surface_distance_prtogt_liver',
                         'surface_distance_prtogt_mean',
                         'surface_distance_gttopr_spleen', 'surface_distance_gttopr_liver',
                         'surface_distance_gttopr_mean', 'surface_distance_avg_spleen', 'surface_distance_avg_liver',
                         'surface_distance_avg_mean']

        # 'umap_surface_distance_prtogt_spleen','umap_surface_distance_prtogt_liver',
        # 'umap_surface_distance_prtogt_mean',
        # 'umap_surface_distance_gttopr_spleen', 'umap_surface_distance_gttopr_liver',
        # 'umap_surface_distance_gttopr_mean',
        # 'umap_surface_distance_avg_spleen','umap_surface_distance_avg_liver',
        # 'umap_surface_distance_avg_mean'

        self.projects = {"full_bayesian": 0, "hierarchical_quicknat": 1, "MC_dropout_quicknat": 2,
                         "probabilistic_quicknat": 3}
        self.datasets = {"KORA": 0, "NAKO": 1, "UKB": 2}

    def ncc(self, a, v, zero_norm=False):

        a = a.flatten()
        v = v.flatten()

        if zero_norm:

            a = (a - np.mean(a)) / (np.std(a) * len(a))
            v = (v - np.mean(v)) / np.std(v) + 0.0001

        else:

            a = a / (np.std(a) * len(a))
            v = v / np.std(v) + 0.0001

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

        n_ = sample_arr.shape[0]
        m_ = gt_arr.shape[0]

        s_x = sample_arr.shape[1]
        s_y = sample_arr.shape[2]

        e_ss_arr = np.zeros((n_, s_x, s_y))
        for i in range(n_):
            e_ss_arr[i, ...] = pixel_wise_xent(sample_arr[i, ...], mean_seg)

        e_ss = np.mean(e_ss_arr, axis=0)

        e_sy_arr = np.zeros((m_, n_, s_x, s_y))
        for j in range(m_):
            for i in range(n_):
                e_sy_arr[j, i, ...] = pixel_wise_xent(sample_arr[i, ...], gt_arr[j, ...])

        e_sy = np.mean(e_sy_arr, axis=1)

        ncc_list = []
        for j in range(m_):
            ncc_list.append(self.ncc(e_ss, e_sy[j, ...]))

        return np.abs((1 / m_) * sum(ncc_list))

    @staticmethod
    def generalised_energy_distance(sample_arr, gt_arr, nlabels, **kwargs):

        def dist_fct(m1, m2):

            label_range = kwargs.get('label_range', range(nlabels))

            per_label_iou = []
            for lbl in label_range:
                if lbl == 0:
                    continue

                m1_bin = (m1 == lbl) * 1
                m2_bin = (m2 == lbl) * 1

                if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                    per_label_iou.append(1)
                elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                    per_label_iou.append(0)
                else:
                    per_label_iou.append(jc(m1_bin, m2_bin))

            return 1 - (sum(per_label_iou) / nlabels)

        """
        :param sample_arr: expected shape N x X x Y 
        :param gt_arr: M x X x Y
        :return: 
        """

        n_ = sample_arr.shape[0]
        m_ = gt_arr.shape[0]

        d_sy = []
        d_ss = []
        d_yy = []

        for i in range(n_):
            for j in range(m_):
                d_sy.append(dist_fct(sample_arr[i, ...], gt_arr[j, ...]))

        for i in range(n_):
            for j in range(n_):
                d_ss.append(dist_fct(sample_arr[i, ...], sample_arr[j, ...]))

        for i in range(m_):
            for j in range(m_):
                d_yy.append(dist_fct(gt_arr[i, ...], gt_arr[j, ...]))

        return (2. / (n_ * m_)) * sum(d_sy) - (1. / n_ ** 2) * sum(d_ss) - (1. / m_ ** 2) * sum(d_yy)

    def intersection_overlap_per_structure_per_slice(self, samples_):
        iou_per_slice = [0, 0]
        for i in range(samples_.shape[1]):
            samples = samples_[:, i]
            for c in range(self.dataUtils.num_class):
                if c == 0:
                    continue
                inter = (samples[0] == c).astype('int')
                union = (samples[0] == c).astype('int')
                for s in range(1, self.dataUtils.mc_sample):
                    nxt = (samples[s] == c).astype('int')
                    inter = np.multiply(inter, nxt)
                    union = (np.add(union, nxt) > 0).astype('int')
                s_inter, s_union = np.sum(inter), np.sum(union) + 0.0001
                # p_diff = (s_union - s_inter) / s_union
                iou_per_slice[c - 1] += np.divide(s_inter, s_union)
        iou_s = np.divide(iou_per_slice, samples_.shape[1])
        return iou_s

    def intersection_overlap_per_structure(self, samples):
        iou_s = []
        for c in range(self.dataUtils.num_class):
            if c == 0:
                continue
            inter = (samples[0] == c).astype('int')
            union = (samples[0] == c).astype('int')
            for s in range(1, self.dataUtils.mc_sample):
                nxt = (samples[s] == c).astype('int')
                inter = np.multiply(inter, nxt)
                union = (np.add(union, nxt) > 0).astype('int')
            s_inter, s_union = np.sum(inter), np.sum(union)
            if s_inter == 0 and s_union == 0:
                iou_s.append(1)
            elif s_inter > 0 and s_union == 0 or s_inter == 0 and s_union > 0:
                iou_s.append(0)
            else:
                iou_s.append(np.divide(s_inter, s_union))
        return iou_s

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

    def uncertainty_map_surface_distance(self, iou_uncertainty, volume_prediction, mode):
        u_map_dice_surface_distance = []
        if not self.dataUtils.is_uncertainity_check_enabled:
            return np.zeros((self.dataUtils.mc_sample, 2, self.dataUtils.num_class))
        for u_map in iou_uncertainty:
            u_map_dice_surface_distance.append(self.dice_surface_distance_perclass(torch.from_numpy(u_map),
                                                                                   volume_prediction,
                                                                                   mode=mode))
        u_map_dice_surface_distance = np.asarray(u_map_dice_surface_distance)
        return u_map_dice_surface_distance

    def dice_surface_distance_perclass(self, vol_output, ground_truth, no_samples=10, mode='train'):
        surface_distance_perclass = np.zeros((self.dataUtils.num_class, 2))
        if mode == 'train':
            samples = np.random.choice(len(vol_output), no_samples)
            vol_output, ground_truth = vol_output[samples], ground_truth[samples]
        for i in range(self.dataUtils.num_class):
            GT = (ground_truth == i).int()
            Pred = (vol_output == i).int()
            surface_dis = compute_surface_distances(GT.cpu().numpy(), Pred.cpu().numpy(),
                                                    self.dataUtils.target_voxel_dimension)
            avg_surface_distance = compute_surface_overlap_at_tolerance(surface_dis, self.dataUtils.organ_tolerances[i])
            surface_distance_perclass[i] = avg_surface_distance

        return surface_distance_perclass

    def scores_to_csv(self, scores, final=False):
        if final:
            csv_file = os.path.join(self.dataUtils.base_dir, 'reports', self.dataUtils.model_name,
                                    self.dataUtils.dataset,
                                    str(self.dataUtils.mc_sample) + '_' + self.dataUtils.ctime + '_final_report.csv')
        else:
            csv_file = os.path.join(self.dataUtils.base_dir, 'reports', self.dataUtils.model_name,
                                    self.dataUtils.dataset,
                                    str(self.dataUtils.mc_sample) + '_' + self.dataUtils.ctime + '_report.csv')

        CommonUtils.create_csv_file_if_not(csv_file, self.csv_cols)
        df = pd.DataFrame([scores], columns=self.csv_cols, index=None)
        df.to_csv(csv_file, mode='a', header=False, index=None)

    def print_report(self, strin, final=False, only_print=True):
        if final:
            report_file = os.path.join(self.dataUtils.base_dir, 'reports', self.dataUtils.model_name,
                                       self.dataUtils.dataset,
                                       str(self.dataUtils.mc_sample) + '_' + self.dataUtils.ctime + '_final_report.txt')
        else:
            report_file = os.path.join(self.dataUtils.base_dir, 'reports', self.dataUtils.model_name,
                                       self.dataUtils.dataset,
                                       str(self.dataUtils.mc_sample) + '_' + self.dataUtils.ctime + '_report.txt')
        if not only_print:
            CommonUtils.create_file_if_not(report_file)
            report_file_obj = open(report_file, 'a+')
            print(strin, end="\n", file=report_file_obj)
            report_file_obj.close()
        print(strin)
        # if self.notifier is not None:
        #    self.notifier.notify(strin)

    def intermediate_report(self, volume_id, volume_dice_score, volume_dice_surface_distance,
                            iou_s, s_ncc_list, s_ged_list, u_map_dice_surface_distance=None):
        scores_for_csv = []
        project_name, dataset, exp_name = self.dataUtils.project_name, self.dataUtils.dataset, self.dataUtils.exp_name
        scores_for_csv.extend([self.projects[project_name], self.datasets[dataset], exp_name, volume_id,
                               self.dataUtils.mc_sample])
        if self.dataUtils.is_uncertainity_check_enabled:
            scores_for_csv.extend([s_ncc_list[-1][0], s_ged_list[-1]])
            scores_for_csv.extend([iou_s[0], iou_s[1], np.mean(iou_s)])
        else:
            scores_for_csv.extend([-1, -1, -1, -1, -1])

        scores_for_csv.extend([volume_dice_score[1], volume_dice_score[2], np.mean(volume_dice_score[1:])])

        scores_for_csv.extend([volume_dice_surface_distance[1, 1],
                               volume_dice_surface_distance[2, 1],
                               np.mean(volume_dice_surface_distance[1:, 1])])

        scores_for_csv.extend([volume_dice_surface_distance[1, 0],
                               volume_dice_surface_distance[2, 0],
                               np.mean(volume_dice_surface_distance[1:, 0])])

        scores_for_csv.extend([np.mean(volume_dice_surface_distance[1, :]),
                               np.mean(volume_dice_surface_distance[2, :]),
                               np.mean(volume_dice_surface_distance[1:])])

        if u_map_dice_surface_distance is not None:
            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, 1, 1]),
                                   np.mean(u_map_dice_surface_distance[:, 2, 1]),
                                   np.mean(u_map_dice_surface_distance[:, 1:, 1])])

            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, 1, 0]),
                                   np.mean(u_map_dice_surface_distance[:, 2, 0]),
                                   np.mean(u_map_dice_surface_distance[:, 1:, 0])])

            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, 1, :]),
                                   np.mean(u_map_dice_surface_distance[:, 2, :]),
                                   np.mean(u_map_dice_surface_distance[:, 1:])])

        self.scores_to_csv(scores_for_csv)

        if self.dataUtils.is_uncertainity_check_enabled:
            self.print_report('# Intersection over overlap scores per structure per volume.')
            self.print_report(f'Spleen: {iou_s[0]}   |    Liver: {iou_s[1]}  |  <-Mean: {np.mean(iou_s)} \n')

            self.print_report(f'# Normalised Cross Correlation score: \n {s_ncc_list[-1][0]}')
            self.print_report(f'# Generalised Entropy Distance score: \n {s_ged_list[-1]}')

        self.print_report('# Dice Score per structure per volume.')
        self.print_report(f'Spleen: {volume_dice_score[1]}   |    Liver: {volume_dice_score[2]}  |  '
                          f'<-Mean: {np.mean(volume_dice_score[1:])} \n')

        self.print_report('# Surface distance scores per structure per volume.')

        self.print_report(f'GT TO PRED SCORES:: Spleen: {volume_dice_surface_distance[1, 0]}   |   '
                          f'Liver: {volume_dice_surface_distance[2, 0]}  |  '
                          f'<-Mean: {np.mean(volume_dice_surface_distance[1:, 0])}')
        self.print_report(f'PRED TO GT SCORES::  Spleen: {volume_dice_surface_distance[1, 1]}   |   '
                          f'Liver: {volume_dice_surface_distance[2, 1]}  |  '
                          f'<-Mean: {np.mean(volume_dice_surface_distance[1:, 1])}')
        self.print_report(f'  MEAN TO ABOVE  :: Spleen: {np.mean(volume_dice_surface_distance[1, :])}   |   '
                          f'Liver: {np.mean(volume_dice_surface_distance[2, :])}  |  '
                          f'MEAN: {np.mean(volume_dice_surface_distance[1:])} \n')

        if u_map_dice_surface_distance is not None:
            self.print_report('# Uncertainty Sample Surface distance scores.', final=True)
            self.print_report(f'GT TO PRED SCORES:: Spleen: {np.mean(u_map_dice_surface_distance[:, 1, 0])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, 2, 0])}  |  '
                              f'<-Mean: {np.mean(u_map_dice_surface_distance[:, 1:, 0])}')
            self.print_report(f'PRED TO GT SCORES::  Spleen: {np.mean(u_map_dice_surface_distance[:, 1, 1])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, 2, 1])}  |  '
                              f'<-Mean: {np.mean(u_map_dice_surface_distance[:, 1:, 1])}')
            self.print_report(f'  MEAN TO ABOVE  :: Spleen: {np.mean(u_map_dice_surface_distance[:, 1, :])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, 2, :])}  |  '
                              f'MEAN: {np.mean(u_map_dice_surface_distance[:, 1:])} \n')

        self.print_report('------------------------------------------------------------------------------ \n\n')

    def final_report(self, dice_score_arr, surface_distance_arr, iou_score_per_structure_arr, s_ncc_arr, s_ged_arr,
                     u_map_dice_surface_distance=None):

        scores_for_csv = []
        project_name, dataset, exp_name = self.dataUtils.project_name, self.dataUtils.dataset, self.dataUtils.exp_name
        scores_for_csv.extend([self.projects[project_name], self.datasets[dataset], exp_name, 0,
                               self.dataUtils.mc_sample])
        if self.dataUtils.is_uncertainity_check_enabled:
            scores_for_csv.extend([np.mean(s_ncc_arr), np.mean(s_ged_arr)])
            scores_for_csv.extend([np.mean(iou_score_per_structure_arr[:, 0]),
                                   np.mean(iou_score_per_structure_arr[:, 1]),
                                   np.mean(iou_score_per_structure_arr)])
        else:
            scores_for_csv.extend([-1, -1, -1, -1, -1])

        scores_for_csv.extend([np.mean(dice_score_arr[:, 1]), np.mean(dice_score_arr[:, 2]),
                               np.mean(dice_score_arr[:, 1:])])

        scores_for_csv.extend([np.mean(surface_distance_arr[:, 1, 1]), np.mean(surface_distance_arr[:, 2, 1]),
                               np.mean(surface_distance_arr[:, 1:, 1])])
        scores_for_csv.extend([np.mean(surface_distance_arr[:, 1, 0]), np.mean(surface_distance_arr[:, 2, 0]),
                               np.mean(surface_distance_arr[:, 1:, 0])])
        scores_for_csv.extend([np.mean(surface_distance_arr[:, 1]), np.mean(surface_distance_arr[:, 2]),
                               np.mean(surface_distance_arr[:, 1:])])

        if u_map_dice_surface_distance is not None:
            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, :, 1, 1]),
                                   np.mean(u_map_dice_surface_distance[:, :, 2, 1]),
                                   np.mean(u_map_dice_surface_distance[:, :, 1:, 1])])

            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, :, 1, 0]),
                                   np.mean(u_map_dice_surface_distance[:, :, 2, 0]),
                                   np.mean(u_map_dice_surface_distance[:, :, 1:, 0])])

            scores_for_csv.extend([np.mean(u_map_dice_surface_distance[:, :, 1, :]),
                                   np.mean(u_map_dice_surface_distance[:, :, 2, :]),
                                   np.mean(u_map_dice_surface_distance[:, :, 1:])])

        self.scores_to_csv(scores_for_csv, final=True)

        self.print_report('++++++++++++++++++++++++++++ FINAL REPORT +++++++++++++++++++++++++++++++\n', final=True)

        if self.dataUtils.is_uncertainity_check_enabled:
            self.print_report('# Mean intersection over overlap scores.', final=True)
            self.print_report(f' OVERALL MEAN: {np.mean(iou_score_per_structure_arr)} | '
                              f'SPLEEN: {np.mean(iou_score_per_structure_arr[:, 0])} | '
                              f'LIVER: {np.mean(iou_score_per_structure_arr[:, 1])}', final=True)

            self.print_report(f'# Mean Normalised Cross Correlation score:\n {np.mean(s_ncc_arr)}', final=True)
            self.print_report(f'# Mean Generalised Entropy Distance score:\n {np.mean(s_ged_arr)}\n', final=True)

        self.print_report('# Mean Dice Score.', final=True)
        self.print_report(f' SPLEEN: {np.mean(dice_score_arr[:, 1])} | '
                          f'LIVER: {np.mean(dice_score_arr[:, 2])} | '
                          f'OVERALL MEAN: {np.mean(dice_score_arr[:, 1:])}\n', final=True)

        self.print_report('# Mean Surface distance scores.', final=True)
        self.print_report(f' GT TO PRED SCORES:: SPLEEN: {np.mean(surface_distance_arr[:, 1, 0])} | '
                          f'LIVER: {np.mean(surface_distance_arr[:, 2, 0])} | '
                          f'OVERALL MEAN: {np.mean(surface_distance_arr[:, 1:, 0])}', final=True)
        self.print_report(f' PRED TO GT SCORES:: SPLEEN: {np.mean(surface_distance_arr[:, 1, 1])} | '
                          f'LIVER: {np.mean(surface_distance_arr[:, 2, 1])} | '
                          f'OVERALL MEAN: {np.mean(surface_distance_arr[:, 1:, 1])}', final=True)
        self.print_report(f' OVERALL MEAN:: SPLEEN: {np.mean(surface_distance_arr[:, 1])} | '
                          f'LIVER: {np.mean(surface_distance_arr[:, 2])} | '
                          f'MEAN: {np.mean(surface_distance_arr[:, 1:])} \n', final=True)

        if u_map_dice_surface_distance is not None:
            self.print_report('# Mean Uncertanty sample Surface distance scores.', final=True)
            self.print_report(f'GT TO PRED SCORES:: Spleen: {np.mean(u_map_dice_surface_distance[:, :, 1, 0])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, :, 2, 0])}  |  '
                              f'<-Mean: {np.mean(u_map_dice_surface_distance[:, :, 1:, 0])}', final=True)
            self.print_report(f'PRED TO GT SCORES::  Spleen: {np.mean(u_map_dice_surface_distance[:, :, 1, 1])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, :, 2, 1])}  |  '
                              f'<-Mean: {np.mean(u_map_dice_surface_distance[:, :, 1:, 1])}', final=True)
            self.print_report(f'  MEAN TO ABOVE  :: Spleen: {np.mean(u_map_dice_surface_distance[:, :, 1, :])}   |   '
                              f'Liver: {np.mean(u_map_dice_surface_distance[:, :, 2, :])}  |  '
                              f'MEAN: {np.mean(u_map_dice_surface_distance[:, :, 1:])} \n', final=True)

        self.print_report('-------------------------------------------------------------------------\n', final=True)

        self.print_report(
            f'############ {self.dataUtils.model_name} on {self.dataUtils.dataset} report completed ############### \n')

    def save_uncertainty_samples(self, iou_uncertainty, prediction_path, current_volume_id, header):
        if not self.dataUtils.is_uncertainity_check_enabled:
            raise Exception('Uncertainty check is not enabled.')
        for i in range(iou_uncertainty.shape[0]):
            sample = iou_uncertainty[i].astype('float32')
            heat_map_nifti_img = nb.MGHImage(np.squeeze(sample), np.eye(4), header=header)
            sample_path = os.path.join(prediction_path, current_volume_id + str('_samples'))
            self.dataUtils.create_if_not(sample_path)
            nb.save(heat_map_nifti_img,
                    os.path.join(sample_path, current_volume_id + str('_sample_{}.nii.gz'.format(i))))

    def save_uncertainty_heat_map(self, heat_map_arr, prediction_path, current_volume_id, header):
        if self.dataUtils.is_uncertainity_check_enabled:
            self.dataUtils.create_if_not(prediction_path)
            heat_map_arr = torch.cat(heat_map_arr, dim=0)
            heat_map_arr = (heat_map_arr.cpu().numpy()).astype('float32')
            heat_map_nifti_img = nb.MGHImage(np.squeeze(heat_map_arr), np.eye(4), header=header)
            nb.save(heat_map_nifti_img, os.path.join(prediction_path, current_volume_id + str('_uncertainty.nii.gz')))
        else:
            raise Exception('Uncertainty check is not enabled.')

    def save_segmentation_map(self, volume_prediction, prediction_path, current_volume_id, header):
        self.dataUtils.create_if_not(prediction_path)
        volume_prediction_ = (volume_prediction.cpu().numpy()).astype('float32')
        nifti_img = nb.MGHImage(np.squeeze(volume_prediction_), np.eye(4), header=header)
        nb.save(nifti_img, os.path.join(prediction_path, current_volume_id + str('_seg.nii.gz')))

    @abc.abstractmethod
    def evaluate_dice_score(self, prediction_path, load_from_txt_file=True, device=0, logWriter=None,
                            is_train_phase=False):
        pass
