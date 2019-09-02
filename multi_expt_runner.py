from settings import compile_config, Settings
from utils.common_utils import CommonUtils

import threading
import os

thesis_path = '/home/abhijit/Jyotirmay/thesis/my_thesis'
master_settings_path = thesis_path + '/master_setting.ini'

to_train = True
to_eval = not to_train
to_do_nothing = None

start_tensorboard = False

models = {
    'full_bayesian': [('full_bayesian_KORA_v5', to_eval)],
    'hierarchical_quicknat': [('hierarchical_quicknat_KORA_v5-nobn', to_eval)],
    'MC_dropout_quicknat': [('MC_dropout_quicknat_KORA_v5', to_eval)],
    'probabilistic_quicknat': [('probabilistic_quicknat_KORA_v5', to_eval)]
}
datasets = ['KORA']
samples = [10]

utill = CommonUtils()

if start_tensorboard:
    log_directory = "bayesian:./projects/full_bayesian/logs,hierarchical:./projects/hierarchical_quicknat/logs," \
                    "mc_droput:./projects/MC_dropout_quicknat/logs,probabilistic:./projects/probabilistic_quicknat/logs"

    t = threading.Thread(target=utill.launchTensorBoard, args=([log_directory]))
    t.start()

for model_key in models.keys():
    for model, train_or_eval in models[model_key]:
        if train_or_eval is to_do_nothing:
            continue
        for dataset in datasets:
            for sample in samples:
                # if train_or_eval and sample is not 10:
                #     continue
                try:
                    #if dataset == 'NAKO' and sample is 10:
                        #continue
                    Settings.update_system_status_values(master_settings_path, 'DEFAULT', 'project_name',
                                                         utill.strinfify_for_setting(model_key))
                    Settings.update_system_status_values(master_settings_path, 'DEFAULT', 'dataset',
                                                         utill.strinfify_for_setting(dataset))
                    exp_mixin = model.split('_')[-1]
                    Settings.update_system_status_values(master_settings_path, 'DEFAULT', 'exp_mixin',
                                                         utill.strinfify_for_setting(exp_mixin))
                    project_settings_path = os.path.join(thesis_path, 'projects', model_key, 'settings.ini')
                    if train_or_eval:
                        if dataset is not 'KORA':
                            Settings.update_system_status_values(project_settings_path, 'TRAINING', 'use_pre_trained',
                                                                 str(False))
                            continue
                        else:
                            Settings.update_system_status_values(project_settings_path, 'TRAINING', 'use_pre_trained',
                                                                 str(True))
                            Settings.update_system_status_values(project_settings_path, 'TRAINING', 'learning_rate',
                                                                 str(5e-5))
                            Settings.update_system_status_values(project_settings_path, 'TRAINING', 'num_epochs',
                                                                 str(20))
                    elif dataset is not 'KORA':
                        continue

                    Settings.update_system_status_values(project_settings_path, 'EVAL', 'mc_sample', str(sample))

                    aggregated_settings = compile_config(master_settings_path)
                    m = utill.import_module('.run',
                                            f'projects.{aggregated_settings.settings_dict["COMMON"]["project_name"]}')
                    executor = m.Executor(aggregated_settings)

                    common_params = executor.dataUtils.common_params
                    data_params = executor.dataUtils.data_params
                    net_params = executor.dataUtils.net_params
                    train_params = executor.dataUtils.train_params
                    eval_params = executor.dataUtils.eval_params
                    data_config_params = executor.dataUtils.data_config_params
                    eval_params.update(data_config_params)

                    if train_or_eval:
                        executor.train(train_params, common_params, data_params, net_params)
                    else:
                        executor.evaluate(eval_params, net_params, data_params, common_params, train_params)
                except Exception as e:
                    print(e)
                    # utill.setup_notifier()
                    # utill.notify('*'+str(e)+'*')
                    continue

