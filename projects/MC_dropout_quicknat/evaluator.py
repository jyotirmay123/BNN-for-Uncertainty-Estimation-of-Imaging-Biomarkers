import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from interfaces.evaluator_inteface import EvaluatorInterface


class Evaluator(EvaluatorInterface):
    def __init__(self, settings):
        super().__init__(settings)

    def evaluate_dice_score(self, prediction_path, load_from_txt_file=True, device=0, logWriter=None,
                            is_train_phase=False):
        mode = 'train' if is_train_phase else 'eval'
        self.print_report(
            '########### Execution Settings for {0} dataset on {1} model ############## \n'.format(
                self.dataUtils.dataset, self.dataUtils.final_model_file), final=True)
        self.print_report(self.dataUtils.settings.settings_dict, final=True)
        self.print_report(self.dataUtils.mc_sample, final=True)

        batch_size = self.dataUtils.eval_batch_size

        # uncomment here jj
        # with open(self.dataUtils.test_volumes) as file_handle:
        #     volumes_to_use = file_handle.read().splitlines()
        #
        # model = torch.load(self.dataUtils.eval_model_path)
        # # torch.save(model.state_dict(), 'mc_dropout_quicknat.model')
        # cuda_available = torch.cuda.is_available()
        # if cuda_available:
        #     torch.cuda.empty_cache()
        #     model.cuda(device)
        #
        # model.eval()

        volume_dice_score_list, volume_surface_distance_list, iou_score_per_structure_list = [], [], []
        s_ncc_list, s_ged_list = [], []

        print("Evaluating now...")
        # uncomment here jj
        # file_paths = self.dataUtils.load_file_paths(load_from_txt_file=load_from_txt_file,
        #                                             is_train_phase=is_train_phase)

        df = pd.read_csv(
            '/home/abhijit/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_UKB_v2/UKB/10_0.0_report.csv',
            index_col=0)
        with open(
                '/home/abhijit/Jyotirmay/my_thesis/dataset_groups/whole_body_datasets/UKB/evaluated_vols.txt') as file_handle:
            volumes_to_use = file_handle.read().splitlines()
        file_paths = []
        for ix, v in enumerate(volumes_to_use):
            targets = df[df['volume_id'] == v]['target_scan_file'].values
            i1, i2 = None, None
            if len(targets) > 1:
                i1, i2 = df[df['volume_id'] == v]['iou_mean'].values
                if i1 > i2:
                    idx = 0
                else:
                    idx = 1
                target = targets[idx]
            elif len(targets) == 0:
                continue
            else:
                target = targets[0]
            input_file = '/home/abhijit/nas_drive/Data_WholeBody/UKBiobank/body/body_nifti/' + str(v) + '/' + target
            gt_file = None
            file_paths.append([input_file, gt_file])
        print(len(file_paths))
        self.print_report('########### Evaluating {0} dataset on {1} model ############## \n'
                          .format(self.dataUtils.dataset, self.dataUtils.model_name))
        self.print_report('########### Evaluating {0} dataset on {1} model ############## \n'
                          .format(self.dataUtils.dataset, self.dataUtils.model_name), final=True)
        with torch.no_grad():
            for vol_idx, file_path in enumerate(file_paths):
                try:
                    print(file_path)
                    volid_or_mixin = file_path[0].split('/')[-1].split('.')[0]
                    self.print_report('# VOLUME:: ' + volid_or_mixin + '\n')
                    vol_mixin = None
                    # Uncomment here jj
                    # if self.dataUtils.dataset == 'UKB':
                    #     vol_idx = vol_idx // 3
                    #     print(volumes_to_use[vol_idx])
                    #     vol_mixin = volid_or_mixin[-1:]

                    if self.dataUtils.label_dir is None:
                        volume, header = self.dataUtils.volume_load_and_preprocess(file_path)
                        # vol_to_save = volume.copy()
                        if self.dataUtils.dataset == 'UKB':
                            self.dataUtils.save_processed_nibabel_file(volume, header, volumes_to_use[vol_idx])

                    else:
                        volume, labelmap, header, weights, class_weights = self.dataUtils.load_and_preprocess(file_path)
                    continue

                    volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
                    volume = torch.tensor(np.ascontiguousarray(volume)).type(torch.FloatTensor)

                    if self.dataUtils.label_dir is not None:
                        labelmap = torch.tensor(np.ascontiguousarray(labelmap)).type(torch.FloatTensor)

                    volume_prediction, heat_map_arr = [], []
                    iou_s, s_ncc, s_ged = None, None, None
                    iou_uncertainty = np.zeros(
                        (self.dataUtils.mc_sample, volume.shape[0], volume.shape[2], volume.shape[3]))

                    for i in range(0, len(volume), batch_size):
                        batch_x = volume[i: i + batch_size]
                        if cuda_available:
                            batch_x = batch_x.cuda(device)

                        if self.dataUtils.is_uncertainity_check_enabled:
                            batch_uncertainty_outputs = []
                            for mcs in range(self.dataUtils.mc_sample):
                                model.is_training = False
                                out = model.forward(batch_x)
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
                        else:
                            model.is_training = False
                            out = model.forward(batch_x)
                            out = out[2]
                            out = F.softmax(out, dim=1)
                            _, batch_output = torch.max(out, dim=1)

                        volume_prediction.append(batch_output)

                    volume_prediction = torch.cat(volume_prediction)
                    if self.dataUtils.label_dir is not None:
                        volume_dice_score = self.dice_score_perclass(volume_prediction, labelmap.cuda(device),
                                                                     self.dataUtils.num_class, mode=mode)
                        volume_dice_surface_distance = self.dice_surface_distance_perclass(volume_prediction,
                                                                                           labelmap.cuda(device), mode=mode)
                    else:
                        volume_dice_score, volume_dice_surface_distance = None, None

                    if self.dataUtils.is_uncertainity_check_enabled:
                        iou_s = self.intersection_overlap_per_structure(iou_uncertainty)

                        if iou_s[0] < 0.30 or iou_s[1] < 0.30:
                            print('iou:', iou_s)
                            print('skip this vol:', volumes_to_use[vol_idx] + vol_mixin)
                            continue
                        if self.dataUtils.dataset == 'UKB':
                            self.dataUtils.save_processed_nibabel_file(vol_to_save, header, volumes_to_use[vol_idx]+vol_mixin)
                        if self.dataUtils.label_dir is not None:
                            s_ncc = self.variance_ncc_dist(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy())
                            s_ged = self.generalised_energy_distance(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy(), 3)
                        else:
                            s_ncc, s_ged = None, None

                    volume_dice_score_list.append(volume_dice_score)
                    volume_surface_distance_list.append(volume_dice_surface_distance)
                    s_ncc_list.append(s_ncc)
                    s_ged_list.append(s_ged)
                    iou_score_per_structure_list.append(iou_s)

                    if self.dataUtils.is_uncertainity_check_enabled:
                        self.save_uncertainty_samples(iou_uncertainty, prediction_path, volumes_to_use[vol_idx]+vol_mixin, header)
                        self.save_uncertainty_heat_map(heat_map_arr, prediction_path, volumes_to_use[vol_idx]+vol_mixin, header)

                    self.save_segmentation_map(volume_prediction, prediction_path, volumes_to_use[vol_idx]+vol_mixin, header)
                    self.intermediate_report(volumes_to_use[vol_idx] + vol_mixin, volume_dice_score,
                                             volume_dice_surface_distance,
                                             iou_s, s_ncc_list, s_ged_list)

                    if self.dataUtils.label_dir is not None and logWriter:
                        logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                                                  vol_idx)
                except Exception as e:
                    print(e)
                    continue

            # dice_score_arr = np.asarray(volume_dice_score_list)
            # surface_distance_arr = np.asarray(volume_surface_distance_list)
            # iou_score_per_structure_arr = np.asarray(iou_score_per_structure_list)
            # s_ncc_arr = np.asarray(s_ncc_list)
            # s_ged_arr = np.asarray(s_ged_list)
            #
            # self.final_report(dice_score_arr, surface_distance_arr, iou_score_per_structure_arr, s_ncc_arr, s_ged_arr)
            if self.dataUtils.label_dir is not None:
                class_dist = [dice_score_arr[:, c] for c in range(self.dataUtils.num_class)]

                if logWriter:
                    logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
            print("DONE")
