import os, os.path as osp
import xml.etree.ElementTree as ET
from typing import Sequence
import numpy as np
from PIL import Image
from collections import Counter
import logging

from gdet.registries import DATASETS
from .custom_dataset import CustomDataset
from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
logger = logging.getLogger("gdet.datasets.dota")

VOC_DEFAULT_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

@DATASETS.register_module()
class VocDataset(CustomDataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """

    def __init__(self, config, min_size=None, **kwargs):
        super().__init__(config, **kwargs)
        self.cfg = config
        
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


        self.img_subdir = config.get("img_subdir", "JPEGImages")
        self.ann_subdir = config.get("ann_subdir", "Annotations")
        ### background is 0
        bg_offset = config.get("bg_offset", 1)
        self.m_catId2name = {i + bg_offset: cat for i, cat in enumerate(self.dst_classes)}
        self.m_catId2label = {i + bg_offset: self.dst_classes.index(cat) for i, cat in enumerate(self.dst_classes)}
        self.min_size = min_size
        is_voc = config.get("is_voc", False)
        if is_voc == True:
            if 'VOC2007' in self.m_img_dir:
                self.year = 2007
            elif 'VOC2012' in self.m_img_dir:
                self.year = 2012
            else:
                raise ValueError('Cannot infer dataset year from m_img_dir')
        self.version = ""

    def init(self):
        ann_file = self.cfg.ann_file
        if type(ann_file) is str:
            data_infos = self.load_annotations(ann_file)
        elif isinstance(ann_file, Sequence):
            data_infos = []
            for af in ann_file:
                data_infos.extend(self.load_annotations(af))
        else:
            raise ValueError(f"Error type of ann_file: {ann_file}")
        self.m_data_infos = data_infos # self.filter_data_info(data_infos)

        val_inds = self._filter_imgs()
        self.m_data_infos = np.array(self.m_data_infos)[val_inds].tolist()
        self.stats()

    def load_annotations(self, ann_file: "str"):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = []
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                img_ids.append(line)
        img_ids = list(sorted(img_ids))
        for img_id in img_ids:
            filename = f'{self.img_subdir}/{img_id}.jpg'
            xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.m_img_dir, self.img_subdir, '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_info = dict(id=img_id, filename=filename, width=width, height=height)
                    
            ann = self.load_ann_info(img_id, root=root)
            data_info['ann'] = ann
            data_infos.append(data_info)

        return data_infos

    # def _filter_imgs(self, min_size=32):
    #     """Filter images too small or without annotation."""
    #     valid_inds = []
    #     for i, data_info in enumerate(self.m_data_infos):
    #         if min(data_info['width'], data_info['height']) < min_size:
    #             continue
    #         if self.filter_empty_gt:
    #             img_id = data_info['id']
    #             xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
    #             tree = ET.parse(xml_path)
    #             root = tree.getroot()
    #             for obj in root.findall('object'):
    #                 name = obj.find('name').text
    #                 if name in self.dst_classes:
    #                     valid_inds.append(i)
    #                     break
    #         else:
    #             valid_inds.append(i)
    #     return valid_inds

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.m_data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def load_ann_info(self, img_id, **kwargs):
        """Load annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        # img_id = self.m_data_infos[idx]['id']
        root = kwargs.get("root", None)
        if root is None:
            xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
        ids = []
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        name2catid = {v: k for k, v in self.m_catId2name.items()}
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.dst_classes:
                continue
            label = name2catid[name]
            diff_node = obj.find('difficult')
            difficult = int(diff_node.text) if diff_node else 0
            
            id_node = obj.find('oid')
            if id_node is not None:
                ids.append(int(id_node.text))
            else:
                ids.append(0)
            
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            # bboxes_ignore=bboxes_ignore.astype(np.float32),
            # labels_ignore=labels_ignore.astype(np.int64)
        )
        if len(ids):
            ids = np.array(ids)
            ann['ids'] = ids
        else:
            ann['ids'] = np.array([0])
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.m_data_infos[idx]['id']
        xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.dst_classes:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids
    
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

        logger.info(f"Dataset objects dist ({len(self.dst_classes)}): {sort_counter}")
        logger.info(f"CLASS CatId2Names: {self.m_catId2name}")
        logger.info(f"Total images: {len(self.m_data_infos)} Empty labels: {no_label_count}, AngleVersion: {self.version}")
