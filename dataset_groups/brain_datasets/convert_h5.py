"""
Convert to h5 utility.
python utils/convert_h5_orig.py -cfg='/home/abhijit/Jyotirmay/thesis/hquicknat/settings.ini'
"""
import h5py
import numpy as np

from dataset_groups.whole_body_datasets.data_utils import DataUtils


class ConvertH5(DataUtils):
    def __init__(self, settings):
        super().__init__(settings)

    def apply_split(self):
        file_paths = self.load_file_paths(False, False)
        print("Total no of volumes to process : %d" % len(file_paths))
        train_ratio, test_ratio = self.data_split.split(",")
        train_len = int((int(train_ratio) / 100) * len(file_paths))
        train_idx = np.random.choice(len(file_paths), train_len, replace=False)
        test_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
        train_file_paths = [file_paths[i] for i in train_idx]
        test_file_paths = [file_paths[i] for i in test_idx]
        return train_file_paths, test_file_paths

    # TODO: Dynamise it later.
    def _write_h5(self, data, label, class_weights, weights, f, mode):
        if self.processed_extn == '.npz':
            no_labelmap, H, W = label[0].shape
            no_slices = 1

            data = np.expand_dims(data, axis=0)
            data = np.concatenate(data)
            data = data.reshape((-1, H, W))

            label = np.expand_dims(label, axis=0)
            label = np.concatenate(label)
            label = label.reshape((-1, no_labelmap, H, W))

        else:
            no_slices, H, W = data[0].shape
            data = np.concatenate(data).reshape((-1, H, W))
            label = np.concatenate(label).reshape((-1, H, W))

        with h5py.File(f[mode][self.h5_key_for_data], "w") as data_handle:
            data_handle.create_dataset(self.h5_key_for_data, data=data)
        with h5py.File(f[mode][self.h5_key_for_label], "w") as label_handle:
            label_handle.create_dataset(self.h5_key_for_label, data=label)
        with h5py.File(f[mode][self.h5_key_for_weights], "w") as weight_handle:
            weight_handle.create_dataset(self.h5_key_for_weights, data=np.concatenate(weights))
        with h5py.File(f[mode][self.h5_key_for_class_weights], "w") as class_weight_handle:
            class_weight_handle.create_dataset(self.h5_key_for_class_weights,
                                               data=np.concatenate(class_weights).reshape((-1, H, W)))

    def convert_h5(self):
        if self.annotations_root is not None:
            if self.is_pre_processed:
                raise Exception('Manual annotations are not pre_processed, but is_pre_processed is True!!!')
            self.merge_annotations()
        # Data splitting
        if self.data_split:
            train_file_paths, test_file_paths = self.apply_split()
        elif self.train_volumes and self.test_volumes:
            train_file_paths = self.load_file_paths(load_from_txt_file=True, is_train_phase=True)
            test_file_paths = self.load_file_paths(load_from_txt_file=True, is_train_phase=False)
        else:
            raise ValueError('You must either provide the split ratio or a train, train dataset list')

        print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))

        # Preparing data files to store data.
        f = self.prepare_h5_file_dictionary()

        # loading,pre-processing and writing train data
        print("===Train data===")
        data_train, label_train, weights_train, class_weights_train = self.load_dataset(train_file_paths)

        self._write_h5(data_train, label_train, class_weights_train, weights_train, f, mode='train')

        # loading,pre-processing and writing test data
        print("===Test data===")
        data_test, label_test, weights_test, class_weights_test = self.load_dataset(test_file_paths)

        self._write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='test')
