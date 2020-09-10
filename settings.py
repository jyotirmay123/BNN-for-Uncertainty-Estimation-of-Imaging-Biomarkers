import ast
import configparser
from configparser import ConfigParser
from collections.abc import Mapping
import json


class ConfigParserWithUpdates(ConfigParser):
    def __init__(self, pre_config_dict=None):
        super().__init__()
        if pre_config_dict is not None:
            self.pre_config = pre_config_dict['COMMON']
            for k in self.pre_config.keys():
                if k in ['project_config_path', 'dataset_config_path']:
                    continue
                self.set('DEFAULT', k, "'" + str(self.pre_config[k]) + "'")

            data_dir_base = "'" + self.pre_config['master_base_dir'] + "/dataset_groups/" + \
                            self.pre_config['dataset_groups'] + "/" + self.pre_config['dataset'] + "'"

            self.set('DEFAULT', 'data_dir_base', data_dir_base)


class Settings(Mapping):
    def __init__(self, setting_file='settings.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(setting_file)
        self.settings_dict = _parse_values(self.config)

    def __getitem__(self, key):
        return self.settings_dict[key]

    def __getattr__(self, key):
        return self.settings_dict[key]

    def __len__(self):
        return len(self.settings_dict)

    def __iter__(self):
        return self.settings_dict.items()

    @staticmethod
    def update_system_status_values(file, section, system, value):
        config = configparser.ConfigParser()
        config.read(file)
        cfgfile = open(file, 'w')
        config.set(section, system, value)
        config.write(cfgfile)
        cfgfile.close()


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def _parse_values(config):
    config_parsed = {}
    for section in config.sections():
        config_parsed[section] = DotDict()
        for key, value in config[section].items():
            config_parsed[section][key] = ast.literal_eval(value)
            if key in ['dataset_config_path', 'project_config_path']:
                data_config = ConfigParserWithUpdates(config_parsed)
                data_config.read(ast.literal_eval(value))
                data_settings_dict = _parse_values(data_config)
                for data_section in data_config.sections():
                    config_parsed[data_section] = data_settings_dict[data_section]
    return config_parsed


# Defining a global scope configuration variable which can be accessed from everywhere in the project.
def compile_config(path='/home/abhijit/Jyotirmay/my_thesis/master_setting.ini', save=True):
    settings = Settings(path)
    if save:
        with open('aggregated_settings.ini', 'w') as configfile:
            configfile.write(json.dumps(settings.settings_dict, indent=4))

    return settings
