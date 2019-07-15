"""
Convert to h5 utility.
python utils/convert_h5.py -cfg='/home/abhijit/Jyotirmay/thesis/punet/settings.ini'
"""

import argparse
import h5py
import numpy as np

from utils.data_utils import DataUtils
from settings import compile_config


class ConvertH5(DataUtils):
    def __init__(self, settings):
        super().__init__(settings)

    def apply_split(self):
        file_paths = self.load_file_paths(self.data_dir, self.label_dir)
        print("Total no of volumes to process : %d" % len(file_paths))
        train_ratio, test_ratio = self.data_split.split(",")
        train_len = int((int(train_ratio) / 100) * len(file_paths))
        train_idx = np.random.choice(len(file_paths), train_len, replace=False)
        test_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
        train_file_paths = [file_paths[i] for i in train_idx]
        test_file_paths = [file_paths[i] for i in test_idx]
        return train_file_paths, test_file_paths

    def _write_h5(self, data, label, f, mode):
        no_slices, H, W = data[0].shape
        with h5py.File(f[mode][self.h5_key_for_data], "w") as data_handle:
            data_handle.create_dataset(self.h5_key_for_data, data=np.concatenate(data).reshape((-1, H, W)))
        with h5py.File(f[mode][self.h5_key_for_label], "w") as label_handle:
            label_handle.create_dataset(self.h5_key_for_label, data=np.concatenate(label).reshape((-1, H, W)))

    def convert_h5(self):
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
        data_train, label_train = self.load_dataset(train_file_paths)

        self._write_h5(data_train, label_train, f, mode='train')

        # loading,pre-processing and writing test data
        print("===Test data===")
        data_test, label_test = self.load_dataset(test_file_paths)

        self._write_h5(data_test, label_test, f, mode='test')


if __name__ == "__main__":
    print("* Start *")
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_file_path', '-cfg', required=True, help='Path to project config file(settings.ini)')

    args = parser.parse_args()
    settings = compile_config(args.settings_file_path)
    convert_h5_object = ConvertH5(settings)
    convert_h5_object.convert_h5()
    print("* Finish *")
