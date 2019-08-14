from utils.common_utils import CommonUtils
import time


class ExtractSettings(CommonUtils):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.ctime = str(time.time())
        self.notifier = None

        # COMMON PROJECT SETTING CONFIGURATIONS
        _common_ = self.settings.COMMON
        self.common_params = _common_

        self.save_model_dir = _common_.save_model_dir
        self.model_name = _common_.model_name
        self.log_dir = _common_.log_dir
        self.device = _common_.device
        self.exp_dir = _common_.exp_dir

        # NETWORK SETTING CONFIGURATIONS
        _network_ = self.settings.NETWORK
        self.net_params = _network_

        self.num_class = _network_.num_class
        self.num_channels = _network_.num_channels
        self.num_filters = _network_.num_filters
        self.kernel_h = _network_.kernel_h
        self.kernel_w = _network_.kernel_w
        self.kernel_c = _network_.kernel_c
        self.stride_conv = _network_.stride_conv
        self.pool = _network_.pool
        self.stride_pool = _network_.stride_pool
        self.se_block = _network_.se_block
        self.drop_out = _network_.drop_out
        self.latent_variables = _network_.latent_variables
        self.sampling_frequency = _network_.sampling_frequency
        self.uncertainty_check = _network_.uncertainty_check
        self.beta_value = _network_.beta_value
        self.gamma_value = _network_.gamma_value

        # TRAINING SETTING CONFIGURATIONS
        _training_ = self.settings.TRAINING
        self.train_params = _training_

        self.learning_rate = _training_.learning_rate
        self.train_batch_size = _training_.train_batch_size
        self.val_batch_size = _training_.val_batch_size
        self.log_nth = _training_.log_nth
        self.num_epochs = _training_.num_epochs
        self.optim_betas = _training_.optim_betas
        self.optim_eps = _training_.optim_eps
        self.optim_weight_decay = _training_.optim_weight_decay
        self.lr_scheduler_step_size = _training_.lr_scheduler_step_size
        self.lr_scheduler_gamma = _training_.lr_scheduler_gamma

        self.use_last_checkpoint = _training_.use_last_checkpoint
        self.use_pre_trained = _training_.use_pre_trained

        # EVAL SETTING CONFIGURATIONS
        _eval_ = self.settings.EVAL
        self.eval_params = _eval_

        self.eval_model_path = _eval_.eval_model_path
        self.eval_batch_size = _eval_.eval_batch_size
        self.histogram_matching = _eval_.histogram_matching
        self.histogram_matching_reference_path = _eval_.histogram_matching_reference_path
        self.is_reduce_slices = _eval_.is_reduce_slices
        self.is_remove_black = _eval_.is_remove_black
        self.voxel_dimension_interpolation = _eval_.voxel_dimension_interpolation
        self.target_voxel_dimension = _eval_.target_voxel_dimension
        self.save_predictions_dir = _eval_.save_predictions_dir
        self.is_uncertainity_check_enabled = _eval_.is_uncertainity_check_enabled
        self.mc_sample = _eval_.mc_sample

        # DEFAULT PROJECT SETTING CONFIGURATIONS
        self.base_dir = _eval_.base_dir
        self.exp_name = _eval_.exp_name
        self.final_model_file = _eval_.final_model_file
        self.pre_trained_path = _eval_.pre_trained_path
        self.dataset = _eval_.dataset
        # self.dataset_config_path = _eval_.dataset_config_path

        # DATA CONFIGURATIONS FROM DATASET CONFIG
        _data_ = self.settings.DATA
        self.data_params = _data_

        self.is_h5_processing = _data_.is_h5_processing
        self.h5_data_dir = _data_.h5_data_dir
        self.h5_train_data_file = _data_.h5_train_data_file
        self.h5_train_label_file = _data_.h5_train_label_file
        self.h5_train_weights_file = _data_.h5_train_weights_file
        self.h5_train_class_weights_file = _data_.h5_train_class_weights_file
        self.h5_test_data_file = _data_.h5_test_data_file
        self.h5_test_label_file = _data_.h5_test_label_file
        self.h5_test_weights_file = _data_.h5_test_weights_file
        self.h5_test_class_weights_file = _data_.h5_test_class_weights_file
        self.h5_volume_name_extractor = _data_.h5_volume_name_extractor
        self.labels = _data_.labels
        self.excluded_volumes = []  # _data_.excluded_volumes

        # DATA CONFIG FROM DATASET CONFIG
        _data_config_ = self.settings.DATA_CONFIG
        self.data_config_params = _data_config_

        self.data_dir = _data_config_.data_dir
        self.annotations_root = _data_config_.annotations_root
        self.label_dir = _data_config_.label_dir
        self.train_volumes = _data_config_.train_volumes
        self.test_volumes = _data_config_.test_volumes
        self.orientation = _data_config_.orientation
        self.data_split = _data_config_.data_split
        self.modality = _data_config_.modality
        self.is_pre_processed = _data_config_.is_pre_processed
        self.multi_label_available = _data_config_.multi_label_available
        self.no_of_masks_per_slice = _data_config_.no_of_masks_per_slice
        self.processed_data_dir = _data_config_.processed_data_dir
        self.processed_label_dir = _data_config_.processed_label_dir
        self.processed_extn = _data_config_.processed_extn

        # DATA EVAL CONFIGURATIONS FROM DATASET CONFIG
        _data_eval_config_ = self.settings.DATA_EVAL_CONFIG
        self.data_eval_config_params = _data_eval_config_

        self.organ_tolerances = _data_eval_config_.organ_tolerances

        # DATA FETCH CONFIGURATIONS FROM DATASET CONFIG
        _data_fetch_configurations_ = self.settings.DATA_FETCH_CONFIGURATIONS
        self.data_fetch_configuration_params = _data_fetch_configurations_

        self.modality_map = eval(_data_fetch_configurations_.__modality_map__)
        self._data_file_path_ = _data_fetch_configurations_.__data_file_path__
        self._label_file_path_ = _data_fetch_configurations_.__label_file_path__
        self.target_dim = _data_fetch_configurations_.__target_dimension__

        # DEFAULT DATASET CONFIGURATIONS FROM DATASET CONFIG
        self.data_dir_base = _data_fetch_configurations_.data_dir_base
