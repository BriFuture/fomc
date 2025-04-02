import numpy as np
import logging
from collections import Counter
from terminaltables import AsciiTable

from torch.utils.data import Dataset

logger = logging.getLogger("gdet.datasets")

class CustomDataset(Dataset):
    def __init__(self, config: "dict", *args, **kwargs):
        super().__init__()
        self.data_infos: "list"
        self.m_data_infos: "list" = []
        self.filter_empty_gt=True
        self.m_transforms = kwargs.get("transforms", None)
        self.test_mode = False
        # test_mode is deped, use Test*** dataset instead
        self.m_catId2label = {}
        self.m_catId2name = {}

        self.dst_classes = config.dst_classes[:]
        self.m_img_dir = config.img_dir
    
    @property
    def img_prefix(self):
        return self.m_img_dir

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def pre_pipeline(self, results, ip_idx=0):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.m_img_dir
        # results['seg_prefix'] = self.seg_prefix
        # results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields']  = []
        # results['transform'] = {}
        # results['addtion']     = ip_idx >= 1
        
    def __len__(self):
        return len(self.m_data_infos)   

    def get_ann_info(self, idx):
        return self.m_data_infos[idx]['ann']
    
    def stats(self):
        pass

    def prepare_train_img(self, idx):
        pass
        
    def __getitem__(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        count = 0
        while count < 100:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                count += 1
                continue
            return data