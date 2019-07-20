import ast
import configparser
from collections.abc import Mapping


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
            if key == 'dataset_config_path':
                data_config = configparser.ConfigParser()
                data_config.read(ast.literal_eval(value))
                #  Calling thefunction recursively to read data directory settings from its config file.
                data_settings_dict = _parse_values(data_config)
                for data_section in data_config.sections():
                    config_parsed[data_section] = data_settings_dict[data_section]
    return config_parsed


# Defining a global scope configuration variable which can be accessed from everywhere in the project.
def compile_config(path='/home/abhijit/Jyotirmay/thesis/hquicknat/settings.ini'):
    settings = Settings(path)
    return settings
