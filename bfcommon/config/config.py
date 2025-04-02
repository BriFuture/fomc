from .base_config import Config as BaseConfig, import_modules_from_strings, RESERVED_KEYS

from types import FunctionType, ModuleType
from copy import copy, deepcopy
from .configdict import ConfigDict

class Config(BaseConfig):
    """包含一些元信息"""
    def __init__(self, cfg_dict: "dict"=None, cfg_text: "str"=None, filename: "str"=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        object.__setattr__(self, '_cfg_dict', ConfigDict(cfg_dict))
        object.__setattr__(self, '_filename', filename)
        self._cfg_dict: "ConfigDict"
        self._filename: "str"
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        object.__setattr__(self, '_text', text)

    def freeze(self):
        """freeze 后，所有 item 都无法设置（原版是存在的key-value可以设置）"""
        self._cfg_dict.set_immutable(True)

    def defrost(self):
        self._cfg_dict.set_immutable(False)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
            value.set_group(name)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
            value.set_group(name)
        self._cfg_dict.__setitem__(name, value)

    def get(self, k, dv = None):
        return self._cfg_dict.get(k, dv)
    
    @staticmethod
    def construct_dict(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        ## bsf.c if value is function type, remove it
        remove_keys = []
        for k, v in cfg_dict.items():
            if callable(v):
                remove_keys.append(k)
        for k in remove_keys:
            cfg_dict.pop(k, None)
            
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return cfg_dict, cfg_text
    
    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        cfg_dict, cfg_text = Config.construct_dict(filename, use_predefined_variables, import_custom_modules)

        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)
    
    def merge_from_dict(self, options: "dict"):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        new_cfg = Config._merge_a_into_b(option_cfg_dict, cfg_dict)
        super(Config, self).__setattr__('_cfg_dict', new_cfg)

    def clone(self):
        _cfg = Config()
        ## bsf.c skip function or module
        for k, v in self.items():
            if isinstance(v, (FunctionType, ModuleType)):
                continue
            setattr(_cfg, k, deepcopy(v))
        return deepcopy(_cfg)

