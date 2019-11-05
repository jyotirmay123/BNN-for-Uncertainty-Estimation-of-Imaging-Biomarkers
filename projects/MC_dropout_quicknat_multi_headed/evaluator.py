import numpy as np
import torch
import torch.nn.functional as F

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

        with open(self.dataUtils.test_volumes) as file_handle:
            volumes_to_use = file_handle.read().splitlines()

        model = torch.load(self.dataUtils.eval_model_path)
        # torch.save(model.state_dict(), 'mc_dropout_quicknat.model')
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.empty_cache()
            model.cuda(device)

        model.eval()

        volume_dice_score_list, volume_surface_distance_list, iou_score_per_structure_list = [], [], []
        s_ncc_list, s_ged_list = [], []

        print("Evaluating now...")
        file_paths = self.dataUtils.load_file_paths(load_from_txt_file=load_from_txt_file,
                                                    is_train_phase=is_train_phase)
        self.print_report('########### Evaluating {0} dataset on {1} model ############## \n'
                          .format(self.dataUtils.dataset, self.dataUtils.model_name))
        self.print_report('########### Evaluating {0} dataset on {1} model ############## \n'
                          .format(self.dataUtils.dataset, self.dataUtils.model_name), final=True)
        with torch.no_grad():
            for vol_idx, file_path in enumerate(file_paths):
                print(file_path)
                self.print_report('# VOLUME:: ' + file_path[0].split('/')[-1].split('.')[0] + '\n')
                volume, labelmap, header, weights, class_weights = self.dataUtils.load_and_preprocess(file_path)

                volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
                volume = torch.tensor(np.ascontiguousarray(volume)).type(torch.FloatTensor)
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
                        out = F.softmax(out, dim=1)
                        _, batch_output = torch.max(out, dim=1)

                    volume_prediction.append(batch_output)

                volume_prediction = torch.cat(volume_prediction)
                volume_dice_score = self.dice_score_perclass(volume_prediction, labelmap.cuda(device),
                                                             self.dataUtils.num_class, mode=mode)
                volume_dice_surface_distance = self.dice_surface_distance_perclass(volume_prediction,
                                                                                   labelmap.cuda(device), mode=mode)
                if self.dataUtils.is_uncertainity_check_enabled:
                    iou_s = self.intersection_overlap_per_structure(iou_uncertainty)
                    s_ncc = self.variance_ncc_dist(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy())
                    s_ged = self.generalised_energy_distance(iou_uncertainty, labelmap.unsqueeze(dim=0).numpy(), 3)

                volume_dice_score_list.append(volume_dice_score)
                volume_surface_distance_list.append(volume_dice_surface_distance)
                iou_score_per_structure_list.append(iou_s)
                s_ncc_list.append(s_ncc)
                s_ged_list.append(s_ged)

                if self.dataUtils.is_uncertainity_check_enabled:
                    self.save_uncertainty_samples(iou_uncertainty, prediction_path, volumes_to_use[vol_idx], header)
                    self.save_uncertainty_heat_map(heat_map_arr, prediction_path, volumes_to_use[vol_idx], header)

                self.save_segmentation_map(volume_prediction, prediction_path, volumes_to_use[vol_idx], header)
                self.intermediate_report(volumes_to_use[vol_idx], volume_dice_score, volume_dice_surface_distance,
                                         iou_s, s_ncc_list, s_ged_list)

                if logWriter:
                    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                                              vol_idx)

            dice_score_arr = np.asarray(volume_dice_score_list)
            surface_distance_arr = np.asarray(volume_surface_distance_list)
            iou_score_per_structure_arr = np.asarray(iou_score_per_structure_list)
            s_ncc_arr = np.asarray(s_ncc_list)
            s_ged_arr = np.asarray(s_ged_list)

            self.final_report(dice_score_arr, surface_distance_arr, iou_score_per_structure_arr, s_ncc_arr, s_ged_arr)

            class_dist = [dice_score_arr[:, c] for c in range(self.dataUtils.num_class)]

            if logWriter:
                logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
            print("DONE")
