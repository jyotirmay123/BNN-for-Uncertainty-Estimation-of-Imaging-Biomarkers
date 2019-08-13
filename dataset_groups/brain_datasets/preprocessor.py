import numpy as np
import nibabel as nb
import scipy.interpolate as si
import operator
import glob
import nrrd
import os
from utils.extract_settings import ExtractSettings


class PreProcess(ExtractSettings):

    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        self._orientation_transform_fs_ = {
            "COR": lambda x: x.transpose((0, 2, 1)),
            "AXI": lambda x: x,
            "SAG": lambda x: x.transpose((2, 1, 0)),
        }
        self.target_orientation = [[0, 1], [1, 1], [2, 1]]
        self.skip_Frame = 1
        self.volume_interpolation_method = 'linear'
        self.labelmap_interpolation_method = 'nearest'

    @staticmethod
    def nrrd_reader(file_path):
        _nrrd = nrrd.read(file_path)
        data = _nrrd[0]
        header = _nrrd[1]
        return data, header

    def merge_annotations(self):
        paths = glob.glob(self.annotations_root + '/*')
        classes = self.labels[1:]  # Excluding background class.

        for p in paths:
            id_ = p.split('/')[-1]
            print("# Manual annotations aggregator for volume: " + id_)
            try:
                annotations = glob.glob(p + '/**')
                if len(annotations) > len(classes):
                    print('skipped')
                    self.excluded_volumes.append(id_)
                    continue

                data_ = {c.lower(): None for c in classes}

                for a in annotations:
                    data, header = PreProcess.nrrd_reader(a)

                    if 'SPLEEN' in a.upper():
                        data_['spleen'] = data
                    elif 'LIVER' in a.upper():
                        data_['liver'] = np.multiply(2, data)

                if data_['spleen'] is None or data_['liver'] is None:
                    self.excluded_volumes.append(id_)
                    print('skipped')
                    continue
                print(data_['spleen'].shape, data_['liver'].shape)
                merged_annotations = np.add(data_['spleen'], data_['liver'])
                img = nb.Nifti1Image(merged_annotations, np.eye(4))
                self.create_if_not(self.label_dir)
                filename = os.path.join(self.label_dir, id_ + self.processed_extn)
                nb.save(img, filename)
            except Exception as e:
                print(e)
                self.excluded_volumes.append(id_)

    def reorient(self, volume, labelmap, header):
        target_orientation = self.target_orientation
        source_orientation = nb.orientations.io_orientation(header.get_best_affine())
        transformation_mat = nb.orientations.ornt_transform(source_orientation, target_orientation)
        volume = nb.orientations.apply_orientation(volume, transformation_mat)
        labelmap = nb.orientations.apply_orientation(labelmap, transformation_mat)
        return volume, labelmap

    def do_interpolate(self, source, steps, is_label=False):
        x, y, z = [steps[k] * np.arange(source.shape[k]) for k in range(3)]
        if is_label:
            method = self.labelmap_interpolation_method
        else:
            method = self.volume_interpolation_method

        f = si.RegularGridInterpolator((x, y, z), source, method=method)

        dx, dy, dz = self.target_voxel_dimension
        new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]
        new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))
        return f(new_grid)

    @staticmethod
    def find_nearest(source_num_arr, target_array=np.arange(0, 500, 16)):
        array = np.asarray(target_array)
        idxs = [(np.abs(array - source_num)).argmin() for source_num in source_num_arr]
        return [array[idx + 1] for idx in idxs]

    def post_interpolate(self, volume, labelmap, target_shape):
        volume = self.do_cropping(volume, target_shape)
        labelmap = self.do_cropping(labelmap, target_shape)
        current_shape = volume.shape
        intended_shape_deficit = target_shape - np.asarray(current_shape)

        paddings = [tuple(
            np.array([np.ceil((pad_tuples / 2) - pad_tuples % 2), np.floor((pad_tuples / 2) + pad_tuples % 2)]).astype(
                'int32')) for pad_tuples in intended_shape_deficit]
        paddings = tuple(paddings)

        volume = np.pad(volume, paddings, mode='constant')
        labelmap = np.pad(labelmap, paddings, mode='constant')

        return volume, labelmap

    @staticmethod
    def do_cropping(source_num_arr, bounding):
        start = list(map(lambda a, da: a // 2 - da // 2, source_num_arr.shape, bounding))
        end = list(map(operator.add, start, bounding))
        for i, val in enumerate(zip(start, end)):
            if val[0] < 0:
                start[i] = 0
                end[i] = source_num_arr.shape[i]
        slices = tuple(map(slice, tuple(start), tuple(end)))
        return source_num_arr[slices]

    def rotate_orientation(self, volume, labelmap):

        f_orient = self._orientation_transform_fs_.get(self.orientation, 'not found!')

        if f_orient == 'not found!':
            raise ValueError("Invalid value for orientation. Pleas see help")

        return f_orient(volume), f_orient(labelmap)

    def reduce_slices(self, volume, labelmap):
        """
        This function removes the useless black slices from the start and end. And then selects every even numbered frame.
        """
        no_slices, H, W = volume.shape
        l_slice, l_H, l_W = labelmap.shape

        if len(np.unique(labelmap)) != len(self.labels):
            raise Exception('all class are not present', np.unique(labelmap))

        if no_slices != l_slice or H != l_H or W != l_W:
            raise Exception('original and segmentation shape does not match')

        mask_vector = np.zeros(no_slices, dtype=int)
        mask_vector[::3], mask_vector[1::3], mask_vector[2::3] = 1, 1, 0
        mask_vector[:self.skip_Frame], mask_vector[-self.skip_Frame:-1] = 0, 0

        data_reduced = np.compress(mask_vector, volume, axis=0).reshape(-1, H, W)
        labels_reduced = np.compress(mask_vector, labelmap, axis=0).reshape(-1, H, W)

        return data_reduced, labels_reduced

    @staticmethod
    def remove_black(volume, labelmap):
        clean_data, clean_labels = [], []
        for i, frame in enumerate(labelmap):
            unique, counts = np.unique(frame, return_counts=True)
            if counts[0] / sum(counts) < .99:
                clean_labels.append(frame)
                clean_data.append(volume[i])
        return np.array(clean_data), np.array(clean_labels)

    def hist_match(self, volume):
        template_file = nb.load(self.histogram_matching_reference_path)
        template = template_file.get_fdata()
        oldshape = volume.shape
        source = volume.ravel()
        template = template.ravel()

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    @staticmethod
    def normalise_data(volume):
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        return volume

    @staticmethod
    def estimate_weights_mfb(labels):
        class_weights = np.zeros_like(labels)
        unique, counts = np.unique(labels, return_counts=True)
        median_freq = np.median(counts)
        weights = np.zeros(len(unique))
        for i, label in enumerate(unique):
            class_weights += (median_freq // counts[i]) * np.array(labels == label)
            weights[int(label)] = median_freq // counts[i]

        grads = np.gradient(labels)
        edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
        class_weights += 2 * edge_weights
        return class_weights, weights

    @staticmethod
    def estimate_weights_per_slice(labels):
        weights_per_slice = []
        for slice_ in labels:
            unique, counts = np.unique(slice_, return_counts=True)
            median_freq = np.median(counts)
            weights = np.zeros(3)
            for i, label in enumerate(unique):
                weights[int(label)] = median_freq // counts[i]
            weights_per_slice.append(weights)

        return np.array(weights_per_slice)