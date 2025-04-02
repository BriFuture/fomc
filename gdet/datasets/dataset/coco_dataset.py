import os.path as osp
import numpy as np
import pickle
import tempfile
import torch
import copy
from collections import Counter

from pycocotools.coco import COCO
import logging
from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from gdet.registries import DATASETS
from .custom_dataset import CustomDataset

__all__ = ["CocoDataset"]

logger = logging.getLogger("gdet.datasets")

COCO_DEFAULT_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

@DATASETS.register_module()    
class CocoDataset(CustomDataset):
    def __init__(self, cfg, transforms=None) -> None:
        """
        cfg: dataset specified
        """
        super().__init__(cfg, transforms=transforms)
        self.cfg = cfg

    def init(self):
        data_infos = self.load_annotations(self.cfg.ann_file)
        self.m_data_infos = data_infos # self.filter_data_info(data_infos)
        val_inds = self._filter_imgs()
        self.m_data_infos = np.array(self.m_data_infos)[val_inds].tolist()
        self.stats()
    
    def load_annotations(self, ann_file: "str"):
        coco = COCO(ann_file)
        logger.info(f"Loading coco ann file: {ann_file}")
        self.m_coco: "COCO"  = coco
        ### bsf.c original all dataset
        # self.m_cat_ids = coco.getCatIds()
        # cat2label = {cat_id['name']: i for i, cat_id in coco.cats.items()}
        catId2label = {}
        catId2name = {}
        for idx, cat_id in coco.cats.items():
            if cat_id['name'] in self.dst_classes:
                catId2label[idx] = self.dst_classes.index(cat_id['name']) + 1 
                catId2name[idx] = cat_id['name']
        self.m_catId2label = catId2label
        ### bsf.c load only needed classes
        self.m_catId2name = catId2name
        self.m_cat_ids = list(self.m_catId2label.keys())

        self.m_img_ids = coco.getImgIds()
        ### bsf.c read img infos
        data_infos = []
        for ni, i in enumerate(self.m_img_ids):
            info = coco.imgs[i]
            info['filename'] = info['file_name']
            img_info = {
                "filename":info['file_name'],
                "height": info['height'],
                "width": info['width'],
                "id": info['id'],
            }
            data_infos.append(img_info)
        return data_infos
    
    def get_mapped_indicator(self, base_category_indicator):
        base_indicator = base_category_indicator[:]
        # label2catId = {v: k for k, v in self.m_catId2label.items()}
        # mapped_indicator = [True] * len(base_indicator)
        # for k, v in label2catId.items():
        #     mapped_indicator[v] = base_indicator[k]
        catId2class = {}
        for k, v in self.m_catId2label.items():
            catId2class[k] = (self.m_catId2name[k], base_indicator[v])
        return catId2class
    # def filter_data_info(self, data_infos):
    #     coco: "COCO" = self.m_coco
    #     new_data_infos = []
    #     for idx in range(len(data_infos)):
    #         dinfo = data_infos[idx]
    #         img_id = dinfo['id']
    #         ann_ids = coco.getAnnIds(imgIds=[img_id])
    #         ann_info = coco.loadAnns(ann_ids)
    #         new_data_infos.append(dinfo)
    #     return new_data_infos
    
    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        coco: "COCO" = self.m_coco
        cur_data_info = copy.deepcopy(self.m_data_infos[idx])
        img_id = cur_data_info['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        info = self._parse_ann_info(cur_data_info, ann_info)
        info['ids'] = [img_id]
        return info
    
    def prepare_train_img(self, idx):
        img_info: "ImageInfo" = self.m_data_infos[idx]
        ann_info: "AnnInfo" = self.get_ann_info(idx)
        if ann_info is None:
            return None
        results = DataBatchItems(img_info=img_info, ann_info=ann_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        if self.m_transforms is not None:
            results = self.m_transforms(results)
        results: "DataTransedItems"
        return results

    def __getitem__(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
                
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            area = ann['area']
            if area <= 0 or w < 1 or h < 1:
                continue
            catId: int = ann['category_id']
            if catId not in self.m_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                label = self.m_catId2label[catId]
                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            # gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            # gt_labels = np.array([], dtype=np.int64)
            return None

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            # bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map
        )
        # if (gt_labels > 80).any():
        #     print(gt_labels)
        return ann
    
    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        coco = self.m_coco
        img_id = self.m_data_infos[idx]['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        coco = self.m_coco
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.m_cat_ids):
            ids_in_cat |= set(coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.m_data_infos):
            img_id = self.m_img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.m_img_ids = valid_img_ids
        return valid_inds
    
    def stats(self):
        ### 对 data info 进行统计
        counter = Counter()
        no_label_count = 0
        label2catId = {v: k for k , v in self.m_catId2label.items()}
        for i in range(len(self.m_data_infos)):
            ann_info = self.get_ann_info(i)
            labels = ann_info['labels']
            if labels is None:
                no_label_count += 1
                continue
            for l in labels:
                catId = label2catId[l]
                name = self.m_catId2name[catId]
                counter[name] += 1
        ### sort by label
        sort_counter = {}
        for c in self.dst_classes:
            sort_counter[c] = counter[c]

        logger.info(f"Dataset objects dist ({len(self.dst_classes)}): {sort_counter}")
        logger.info(f"CLASS CatId2Names: {self.m_catId2name}")
        logger.info(f"CLASS Cat2Indexes: {self.m_catId2label}")
        idx2name = {}
        for catId, idx in self.m_catId2label.items():
            idx2name[idx] = self.m_catId2name[catId]
        logger.info(f"CLASS indexes2Name: {idx2name}")
        logger.info(f"Empty labels: {no_label_count}")    

