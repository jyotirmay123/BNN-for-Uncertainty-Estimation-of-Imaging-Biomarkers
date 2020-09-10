import argparse
import os
import shutil
from utils.common_utils import CommonUtils
from settings import compile_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train, eval and converth5')
    parser.add_argument('--settings_file_path', '-cfg', required=True, help='Path to project config file(master_settings.ini)')
    args = parser.parse_args()

    settings = compile_config(args.settings_file_path)

    if args.mode == 'converth5':
        m = CommonUtils.import_module('.convert_h5',
                                      f'dataset_groups.{settings.settings_dict["COMMON"]["dataset_groups"]}')
        convert_h5_object = m.ConvertH5(settings)
        convert_h5_object.convert_h5()
    else:
        m = CommonUtils.import_module('.run',
                                      f'projects.{settings.settings_dict["COMMON"]["project_name"]}')
        Executor = m.Executor
        executor = Executor(settings)

        common_params = executor.dataUtils.common_params
        data_params = executor.dataUtils.data_params
        net_params = executor.dataUtils.net_params
        train_params = executor.dataUtils.train_params
        eval_params = executor.dataUtils.eval_params
        data_config_params = executor.dataUtils.data_config_params
        eval_params.update(data_config_params)

        print(net_params)


        if args.mode == 'train':
            executor.train(train_params, common_params, data_params, net_params)
        elif args.mode == 'eval':
            executor.evaluate(eval_params, net_params, data_params, common_params, train_params)
        elif args.mode == 'clear':
            shutil.rmtree(os.path.join(common_params.exp_dir, train_params.exp_name))
            print("Cleared current experiment directory successfully!!")
            shutil.rmtree(os.path.join(common_params.log_dir, train_params.exp_name))
            print("Cleared current log directory successfully!!")

        elif args.mode == 'clear-all':
            executor.delete_contents(common_params.exp_dir)
            print("Cleared experiments directory successfully!!")
            executor.delete_contents(common_params.log_dir)
            print("Cleared logs directory successfully!!")
        else:
            raise ValueError('Invalid value for mode. only support values are train, eval and clear')
    print("* Finish *")
