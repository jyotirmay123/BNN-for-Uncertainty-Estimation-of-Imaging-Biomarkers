from settings import compile_config, Settings
from run import Executor
from utils.common_utils import CommonUtils

settings_path = '/home/abhijit/Jyotirmay/thesis/hquicknat/settings.ini'

models_with_dropouts = ['"punet_v4_with_dropout"', '"hquicknat_v4_with_dropout"']

models = ['"quicknat_v4"', '"punet_v4"', '"hquicknat_v4"', '"full_bayesian_quicknat"']
datasets = ['"KORA"', '"NAKO"', '"UKB"']
samples = [10, 50, 100]

utill_obj = CommonUtils()
utill_obj.setup_whatsapp_notifier()

for model in models:
    for dataset in datasets:
        for sample in samples:
            try:
                Settings.update_system_status_values(settings_path, 'DEFAULT', 'exp_name', str(model))
                Settings.update_system_status_values(settings_path, 'DEFAULT', 'dataset', str(dataset))
                Settings.update_system_status_values(settings_path, 'EVAL', 'mc_sample', str(sample))

                settings = compile_config(settings_path)
                executor = Executor(settings)

                executor.notifier = utill_obj

                common_params = executor.common_params
                data_params = executor.data_params
                net_params = executor.net_params
                train_params = executor.train_params
                eval_params = executor.eval_params
                data_config_params = executor.data_config_params
                eval_params.update(data_config_params)

                if False:
                    executor.train(train_params, common_params, data_params, net_params)
                else:
                    executor.evaluate(eval_params, net_params, data_params, common_params, train_params)
            except Exception as e:

                print(e)
                utill_obj.whatsapp_notifier('*'+str(e)+'*')
                continue
