# Copyright (c) OpenMMLab. All rights reserved.
from gdet.registries import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class SARDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]
