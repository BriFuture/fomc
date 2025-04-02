import torch
import numpy as np
from bfcommon.config import Config
import os, sys
import os.path as osp
import importlib
import logging
from typing import Sequence

from gdet.structures.configure import ConfigType, DistConfigType
from gdet.parallel import init_dist
# logger = logging.getLogger("gdet.engine.eval")

def load_modules(cfg, hint=False):
    if hasattr(cfg, "modules"):        
        for name, mods in cfg.modules.items():
            if isinstance(mods, Sequence):
                for mod in mods:
                    if mod.endswith(".py"):
                        mod = mod[:-3]
                    mod = mod.replace("/", ".")
                    if mod not in sys.modules:
                        importlib.import_module(mod)
                        if hint:
                            print(f"[Mod] {mod} is being loaded")
            elif type(mods) is str:
                if mod not in sys.modules:
                    importlib.import_module(mods)
                    if hint:
                        print(f"[Mod] {mod} is being loaded")
            else:
                raise ValueError(f"Unsupported Type ({type(mods)}) for {name}: {mods}")

class ExpBase():
    @staticmethod
    def register_config_path():
        CURR_PATH = osp.abspath(osp.dirname(__file__))
        ROOT_PATH = osp.abspath(osp.join(CURR_PATH, "../../",))
        CONFIG_ROOT_PATH = osp.join(ROOT_PATH, "configs")
        Config.PREDEFINED_DATA.update({
            "@model/": osp.join(CONFIG_ROOT_PATH, "model/"),
            # "@dataset/": osp.join(CONFIG_ROOT_PATH, "dataset"),
        })
        
    @staticmethod
    def setup_dist_engine(cfg):
        dist_cfg: "DistConfigType" = cfg.dist
        os_env = os.environ
        if "MASTER_ADDR" not in os_env:
            os_env["MASTER_ADDR"] = dist_cfg.addr
        if "MASTER_PORT" not in os_env:
            os_env["MASTER_PORT"] = str(dist_cfg.port)
        if "WORLD_SIZE" not in os_env:
            os_env['WORLD_SIZE'] = str(dist_cfg.world_size)
        if "RANK" not in os_env:
            os_env['RANK'] = str(dist_cfg.rank)
        init_dist(launcher=dist_cfg.launcher, backend=dist_cfg.backend)

    @staticmethod
    def load_spec_module(config):
        """now modules can be loaded by spec, the whole system can be loaded when need
        """

        load_modules(config.experiment)
        load_modules(config.dataset)
        load_modules(config.model)


    @staticmethod
    def filter_config(cfg: "ConfigType"):
        """remove useless config and make it clean
        """
        cfg.defrost()
        top_level_keys = [
            ### rt
            "train", "val", "test", "weights", "experiment",
            ### model
            "model", 
            ### ds
            "dataset", 
            "dist",
        ]
        new_cfg = {k: cfg[k] for k in top_level_keys if k in cfg}
        additional_keys = set(cfg.keys()) - set(top_level_keys)
        additional_keys = list(filter(lambda x: not x.startswith("_"), additional_keys))
        additional_keys = list(filter(lambda x: not x.isupper(), additional_keys))
        if len(additional_keys):
            print("These keys are not used \n", additional_keys)
        ncfg = Config(new_cfg, filename=cfg.filename)
        ncfg.freeze()
        return ncfg    