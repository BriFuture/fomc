# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp

from collections import Counter
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
import pickle
import logging

from .custom_dataset import CustomDataset
from gdet.core.bbox import obb2poly_np, poly2obb_np
from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from gdet.registries import DATASETS

logger = logging.getLogger("gdet.datasets.dota")
def read_cache(cache_file, cache_version=None):
    if not osp.exists(cache_file):
        return None
    try:
        with open(cache_file, "rb") as f:
            cache_data: dict = pickle.load(f)
    except:
        os.remove(cache_file)
        return None
    data_infos = None
    if cache_data['cache_version'] == cache_version:
        data_infos = cache_data['data']
    print("Load cache version")
    return data_infos
@DATASETS.register_module()
class DOTADataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """

    def __init__(self, config, transforms=None):
        super().__init__(config, transforms=transforms)
        
        difficulty=config.get("difficulty", 100)
        self.version = config.version
        self.difficulty = difficulty
        self.without_ann = False
        self.ann_file = config.get("ann_file", "")
        self.ann_folder = config.ann_folder
        self.ann_cache_version = config.get("cache_version", None)

        ## for few shot specification
        all_classes = config.get("all_classes", None)
        base_classes = config.get("base_classes", None)
        novel_classes = config.get("novel_classes", None)
        dst_class_from_all = config.get("dst_class_from_all", False)
        if dst_class_from_all:
            assert all_classes is not None
            self.dst_classes = all_classes
            
        if all_classes is None:
            self.all_classes = self.dst_classes
        else:
            self.all_classes = all_classes
        if base_classes is None:
            self.base_classes = self.dst_classes
            self.novel_classes = self.dst_classes
        else:
            self.base_classes = base_classes
            self.novel_classes = novel_classes
        ## end few shot
        bg_offset = config.get("bg_offset", 0)
        self.m_catId2name = {i + bg_offset: cat for i, cat in enumerate(self.dst_classes)}
        self.cls_map = {c: i + bg_offset
            for i, c in enumerate(self.dst_classes)
        }  # in mmdet v2.0 label is 0-based
        self.m_catId2label = {i + bg_offset: self.dst_classes.index(cat) for i, cat in enumerate(self.dst_classes)}
        # catId2Name = {i: c for i, c in enumerate(self.dst_classes) }  # in mmdet v2.0 label is 0-based
        # self.m_catId2name = catId2Name
        # self.m_catId2label = {i: i for i, c in enumerate(self.dst_classes)}
        
    def init(self):
        data_infos = self.load_annotations(self.ann_folder, self.ann_cache_version)
        self.m_data_infos = data_infos # self.filter_data_info(data_infos)
        val_inds = self._filter_imgs()
        self.m_data_infos = np.array(self.m_data_infos)[val_inds].tolist()
        if not self.without_ann:
            self.stats()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.m_data_infos)

    def _load_test_phase_anno(self, ann_files):
        data_infos = []
        for ann_file in ann_files:
            data_info = {}
            img_id = osp.split(ann_file)[1][:-4]
            img_name = img_id + '.png'
            data_info['filename'] = img_name
            data_info['ann'] = {}
            data_info['ann']['bboxes'] = []
            data_info['ann']['labels'] = []
            data_infos.append(data_info)
        return data_infos
    
    def _load_train_phase_anno(self, ann_files):
        g_img_id = 1
        data_infos = []
        # im_filename2id = {}
        for ann_file in tqdm(ann_files):
            data_info = {}
            img_id = osp.split(ann_file)[1][:-4]
            img_name = img_id + '.png'
            data_info['filename'] = img_name
            data_info['file_id'] = img_id
            data_info['id'] = g_img_id
            # im_filename2id[img_name] = g_img_id
            g_img_id += 1
            if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                data_info['ann'] = {}
                continue
            ann = self.load_ann_info(ann_file)
            data_info['ann'] = ann
            data_infos.append(data_info)
        return data_infos
    
    def load_ann_info(self, ann_file, **kwargs):
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []
        ids = []
        with open(ann_file) as f:
            s = f.readlines()
        for si in s:
            bbox_info = si.split()
            if len(bbox_info) < 8: 
                continue
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                x, y, w, h, a = poly2obb_np(poly, self.version)
            except:  # noqa: E722
                continue
            cls_name = bbox_info[8]
            difficulty = int(bbox_info[9])
            if cls_name not in self.cls_map:
                continue

            if len(bbox_info) > 10:
                id_item = bbox_info[10]
                assert "id:" in id_item
                ids.append(int(id_item[3:]))
            else:
                ids.append(0)
            label = self.cls_map[cls_name]
            if difficulty > self.difficulty:
                gt_bboxes_ignore.append([x, y, w, h, a])
                gt_labels_ignore.append(label)
                gt_polygons_ignore.append(poly)
            else:
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)
        ann = {}
        if gt_bboxes:
            ann['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            ann['labels'] = np.array(gt_labels, dtype=np.int64)
            ann['polygons'] = np.array(gt_polygons, dtype=np.float32)
        else:
            ann['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels'] = np.array([], dtype=np.int64)
            ann['polygons'] = np.zeros((0, 8), dtype=np.float32)

        if gt_polygons_ignore:
            ann['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(gt_labels_ignore, dtype=np.int64)
            ann['polygons_ignore'] = np.array(gt_polygons_ignore, dtype=np.float32)
        else:
            ann['bboxes_ignore'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels_ignore'] = np.array([], dtype=np.int64)
            ann['polygons_ignore'] = np.zeros((0, 8), dtype=np.float32)
        
        if len(ids):
            ids = np.array(ids)
            ann['ids'] = ids
        else:
            ann['ids'] = np.array([0])

        return ann

    def load_annotations(self, ann_folder: str, cache_version=None):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        ann_files = glob.glob(ann_folder + '/*.txt')
        ann_files = list(sorted(ann_files))
        ann_parent_folder = osp.abspath(osp.join(ann_folder, ".."))
        
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            ann_files = list(sorted(ann_files))
            assert len(ann_files), ann_folder
            data_infos = self._load_test_phase_anno(ann_files)
            self.without_ann = True
        else:
            cache_file = osp.join(ann_parent_folder, f"train_cached_annotations_{self.version}.dat")
            data_infos = None
            if cache_version is not None:
                data_infos = read_cache(cache_file, cache_version)

            if data_infos is None:
                data_infos = self._load_train_phase_anno(ann_files)
                if cache_version is not None:
                    with open(cache_file, "wb") as f:
                        cache_data = {"cache_version": cache_version, "data": data_infos}
                        pickle.dump(cache_data, f)
        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]

        return data_infos
    
    def get_mapped_indicator(self, base_category_indicator):
        return None

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.m_data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


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


    def stats(self):
        counter = Counter()
        no_label_count = 0
        
        for i in range(len(self.m_data_infos)):
            ann_info = self.get_ann_info(i)
            labels = ann_info['labels']
            if labels is None:
                no_label_count += 1
                continue
            for l in labels:
                name = self.m_catId2name[l]
                counter[name] += 1
        ### sort by label
        sort_counter = {}
        for c in self.dst_classes:
            sort_counter[c] = counter[c]

        logger.info(f"Dataset objects dist ({len(sort_counter)}): {sort_counter}")
        assert (sum(sort_counter.values())) > 0, self.ann_file
        logger.info(f"CLASS CatId2Names: {self.m_catId2name}")
        logger.info(f"Total images: {len(self.m_data_infos)} Empty labels: {no_label_count}, AngleVersion: {self.version}")
