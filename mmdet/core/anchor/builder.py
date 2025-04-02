# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from gdet.registries import Registry
# , build_from_cfg

PRIOR_GENERATORS = Registry('Generator for anchors and points')

ANCHOR_GENERATORS = PRIOR_GENERATORS


def build_prior_generator(cfg, default_args=None):
    pg_cfg = cfg.copy()
    pg_type = pg_cfg.pop("type")
    pg_cls = PRIOR_GENERATORS.get(pg_type)
    assert pg_cls is not None, pg_type
    pg = pg_cls(**pg_cfg)
    return pg
    # return build_from_cfg(cfg, PRIOR_GENERATORS, default_args)


def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``build_prior_generator`` ')
    return build_prior_generator(cfg, default_args=default_args)
