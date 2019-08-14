import os
import h5py
import torch
import numpy as np
import nibabel as nb
import torch.utils.data as data
import re
import glob
from dataset_groups.whole_body_datasets.preprocessor import PreProcess


class ImdbData(data.Dataset):
    def __init__(self, X, y, w=None, cw=None, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.cw = cw
        self.transforms = transforms

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])

        if self.cw is not None and self.w is not None:
            weight = torch.from_numpy(self.w[index])
            class_weight = torch.from_numpy(self.cw[index])
            return img, label, weight, class_weight
        if self.w is not None:
            weight = torch.from_numpy(self.w[index])
            return img, label, weight
        if self.cw is not None:
            class_weight = torch.from_numpy(self.cw[index])
            return img, label, class_weight

        return img, label

    def __len__(self):
        return len(self.y)


class DataUtils(PreProcess):
    def __init__(self, settings):
        super().__init__(settings)
        self.h5_key_for_data = 'data'
        self.h5_key_for_label = 'label'
        self.h5_key_for_weights = 'weights'
        self.h5_key_for_class_weights = 'class_weights'

    def prepare_h5_file_dictionary(self):
        self.create_if_not(self.h5_data_dir)
        f = {
            'train': {
                "data": os.path.join(self.h5_data_dir, self.h5_train_data_file),
                "label": os.path.join(self.h5_data_dir, self.h5_train_label_file),
                "weights": os.path.join(self.h5_data_dir, self.h5_train_weights_file),
                "class_weights": os.path.join(self.h5_data_dir, self.h5_train_class_weights_file)
            },
            'test': {
                "data": os.path.join(self.h5_data_dir, self.h5_test_data_file),
                "label": os.path.join(self.h5_data_dir, self.h5_test_label_file),
                "weights": os.path.join(self.h5_data_dir, self.h5_test_weights_file),
                "class_weights": os.path.join(self.h5_data_dir, self.h5_test_class_weights_file)
            }
        }
        return f

    def get_imdb_dataset(self):
        f = self.prepare_h5_file_dictionary()
        data_train = h5py.File(f['train']['data'], 'r')
        label_train = h5py.File(f['train']['label'], 'r')
        weight_train = h5py.File(f['train']['weights'], 'r')
        class_weight_train = h5py.File(f['train']['class_weights'], 'r')

        data_test = h5py.File(f['test']['data'], 'r')
        label_test = h5py.File(f['test']['label'], 'r')
        weight_test = h5py.File(f['test']['weights'], 'r')
        class_weight_test = h5py.File(f['test']['class_weights'], 'r')

        return (ImdbData(data_train[self.h5_key_for_data][()], label_train[self.h5_key_for_label][()],
                         weight_train[self.h5_key_for_weights][()],
                         class_weight_train[self.h5_key_for_class_weights][()]),
                ImdbData(data_test[self.h5_key_for_data][()], label_test[self.h5_key_for_label][()],
                         weight_train[self.h5_key_for_weights][()],
                         class_weight_test[self.h5_key_for_class_weights][()]))

    def load_dataset(self, file_paths):
        print("Loading and preprocessing data...")
        volume_list, labelmap_list, weights_list, class_weights_list = [], [], [], []
        for file_path in file_paths:
            volume_id = eval(self.h5_volume_name_extractor.format(file_path[0]))
            if volume_id in self.excluded_volumes:
                continue
            try:
                volume, labelmap, header, weights, class_weights = self.load_and_preprocess(file_path)

                if self.is_h5_processing:
                    self.save_processed_nibabel_file(volume, header, volume_id)
                    self.save_processed_nibabel_file(labelmap, header, volume_id, True)

                volume_list.append(volume)
                labelmap_list.append(labelmap)
                class_weights_list.append(class_weights)
                weights_list.append(weights)
                print("#", end='', flush=True)
            except Exception as e:
                print(volume_id, e)
                self.excluded_volumes.append(volume_id)
                continue
        # Updating data_config_file as data_related to data_config has been pre-processed now.
        # Settings.update_system_status_values(self.dataset_config_path, 'DATA_CONFIG', 'is_pre_processed', 'True')

        print("100%", flush=True)
        return volume_list, labelmap_list, weights_list, class_weights_list

    def load_and_preprocess(self, file_path):

        volume, labelmap, header = self.load_data(file_path)

        if self.is_pre_processed:
            print('== loading pre-processed data ==')
            volume = self.normalise_data(volume)
            class_weights, weights = self.estimate_weights_mfb(labelmap)
            return volume, labelmap, header, weights, class_weights

        print(' == Pre-processing raw data ==')

        steps = header['pixdim'][1:4]

        volume, labelmap = self.reorient(volume, labelmap, header)

        volume = self.do_interpolate(volume, steps)
        labelmap = self.do_interpolate(labelmap, steps, is_label=True)

        self.target_dim = self.find_nearest(volume.shape) if self.target_dim is None else self.target_dim

        volume, labelmap = self.post_interpolate(volume, labelmap, target_shape=self.target_dim)

        volume, labelmap = self.rotate_orientation(volume, labelmap)

        labelmap = np.moveaxis(labelmap, 2, 0)
        volume = np.moveaxis(volume, 2, 0)

        if self.is_reduce_slices:
            volume, labelmap = self.reduce_slices(volume, labelmap)

        if self.is_remove_black:
            volume, labelmap = self.remove_black(volume, labelmap)

        if self.histogram_matching:
            volume = self.hist_match(volume)

        volume = self.normalise_data(volume)
        class_weights, _ = self.estimate_weights_mfb(labelmap)
        weights = self.estimate_weights_per_slice(labelmap)

        print(volume.shape, labelmap.shape, class_weights.shape, weights.shape)
        return volume, labelmap, header, weights, class_weights

    @staticmethod
    def load_data(file_path):
        volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
        volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
        return volume, labelmap, volume_nifty.header

    @staticmethod
    def load_image_data(file_path, is_multiple_labels_available=False):

        volume = np.load(file_path[0])['np_data'].astype('float64')
        labelmap = None
        if is_multiple_labels_available:
            labelmaps = []
            for l_file in file_path[1]:
                labelmaps.append(np.load(l_file)['np_data'].astype('float64'))
            labelmap = np.asarray(labelmaps)
        else:
            labelmap = np.load(file_path[1])['np_data'].astype('float64')

        return volume, labelmap, None

    def save_processed_nibabel_file(self, volume, header, filename, is_label=False):
        mgh = nb.MGHImage(volume, np.eye(4), header)
        processed_dest_folder = self.processed_data_dir if not is_label else self.processed_label_dir
        self.create_if_not(processed_dest_folder)
        dest_file = os.path.join(processed_dest_folder, filename + self.processed_extn)
        nb.save(mgh, dest_file)
        print('file saved in ' + dest_file)

    def save_nibabel(self, volume, header, filename, vol):
        mgh = nb.MGHImage(volume, np.eye(4), header)
        processed_dest_folder = '/home/abhijit/Jyotirmay/thesis/hquicknat/processeddata/' + vol
        self.create_if_not(processed_dest_folder)
        dest_file = os.path.join(processed_dest_folder, filename + self.processed_extn)
        nb.save(mgh, dest_file)
        print('file saved in ' + dest_file)

    def load_preprocessed_file_paths(self, load_from_txt_file=False, is_train_phase=False):
        if load_from_txt_file:
            volume_txt_file = self.train_volumes if is_train_phase else self.test_volumes
            with open(volume_txt_file) as file_handle:
                volumes_to_use = file_handle.read().splitlines()
        else:
            volumes_to_use = [name for name in os.listdir(self.data_dir)]

        file_paths = [[os.path.join(self.data_dir, vol + self.processed_extn),
                       os.path.join(self.label_dir, vol + self.processed_extn)] for vol in
                      volumes_to_use]

        return file_paths

    # TODO: Reconfigure this function. now, bit dependant on KORA set.
    def load_file_paths(self, load_from_txt_file=False, is_train_phase=False):
        if load_from_txt_file:
            volume_txt_file = self.train_volumes if is_train_phase else self.test_volumes
            with open(volume_txt_file) as file_handle:
                volumes_to_use = file_handle.read().splitlines()
        else:
            volumes_to_use = [name for name in os.listdir(self.data_dir)]

        file_paths = []

        for vol in volumes_to_use:
            try:
                if vol in self.excluded_volumes:
                    print('== {} Volume in Excluded List =='.format(vol))
                    continue

                if self.is_pre_processed:
                    if self.multi_label_available:
                        # Just processing vol names coming after generic pipeline before.
                        vol = vol.split('.')[0]
                        multi_labels = [
                            os.path.join(self.processed_label_dir, vol + '_mask_' + str(mask_id) + self.processed_extn)
                            for mask_id in range(self.no_of_masks_per_slice)]
                        file_paths.append([os.path.join(self.processed_data_dir, vol + self.processed_extn),
                                           multi_labels])
                    else:
                        file_paths.append([os.path.join(self.processed_data_dir, vol + self.processed_extn),
                                           os.path.join(self.processed_label_dir, vol + self.processed_extn)])
                else:
                    data_file_path = self._data_file_path_.format(self.data_dir, vol,
                                                                  self.modality_map[str(self.modality)])
                    label_file_path = self._label_file_path_.format(self.label_dir, vol)

                    files = glob.glob(eval(data_file_path))
                    file_filtration_criterion = np.array([re.findall(r'\d+', f)[-1] for f in files], dtype='int')
                    file_idx = np.where(file_filtration_criterion == file_filtration_criterion.min())
                    file = files[file_idx[0][0]]
                    file = [file]

                    if len(file) is not 0:
                        file_paths.append([file[0], eval(label_file_path)])
                    else:
                        raise Exception('File not found!')
            except Exception as e:
                self.excluded_volumes.append(vol)
                print(vol, e)
                continue

        return file_paths
