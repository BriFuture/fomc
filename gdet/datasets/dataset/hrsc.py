# Copyright (c) OpenMMLab. All rights reserved.
import os, os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
import logging
from PIL import Image
from collections import Counter
from typing import Sequence

from gdet.core.bbox import  obb2poly_np, poly2obb_np
from gdet.registries import DATASETS
from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from .custom_dataset import CustomDataset

logger = logging.getLogger("gdet.datasets.hrsc")

@DATASETS.register_module()
class HRSCDataset(CustomDataset):
    """HRSC dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        classwise (bool): Whether to use all classes or only ship.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    CLASSES = None
    HRSC_CLASS = ('ship', )
    HRSC_CLASSES = ('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                    'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                    'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                    'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                    'Container', 'OXo|--)', 'Car carrier([]==[])',
                    'Hovercraft', 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise',
                    'submarine', 'lute', 'Medical', 'Car carrier(======|',
                    'Ford-class', 'Midway-class', 'Invincible-class')
    HRSC_CLASSES_ID = ('01', '02', '03', '04', '05', '06', '07', '08', '09',
                       '10', '11', '12', '13', '14', '15', '16', '17', '18',
                       '19', '20', '22', '24', '25', '26', '27', '28', '29',
                       '30', '31', '32', '33')
    PALETTE = [
        (0, 255, 0),
    ]
    CLASSWISE_PALETTE = [(220, 20, 60), (119, 11, 32),
                         (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
                         (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                         (100, 170, 30), (220, 220, 0), (175, 116, 175),
                         (250, 0, 30), (165, 42, 42), (255, 77, 255),
                         (0, 226, 252), (182, 182, 255), (0, 82, 0),
                         (120, 166, 157), (110, 76, 0), (174, 57, 255),
                         (199, 100, 0), (72, 0, 118), (255, 179, 240),
                         (0, 125, 92), (209, 0, 151), (188, 208, 182),
                         (0, 220, 176), (255, 99, 164), (92, 0, 73)]

    def __init__(self,
                 config,
                 transforms,
                 **kwargs):
        self.ann_file = config.get("ann_file", "")
        img_subdir = config.get("img_subdir", 'AllImages',)
        self.img_subdir = img_subdir
        ann_subdir = config.get("ann_subdir", 'Annotations',)
        self.ann_subdir = ann_subdir

        self.classwise = config.get("classwise", False)
        self.ann_cache_version = config.get("cache_version", None)
        self.version = config.get("version", 'oc')
        # self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        super(HRSCDataset, self).__init__(config, transforms=transforms, **kwargs)



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
        if self.classwise:
            HRSCDataset.PALETTE = HRSCDataset.CLASSWISE_PALETTE
            catIdPrefix = '1' + '0' * 6
            catId2label = {}
            catId2name = {}
            name2id = {v: k for k, v in config.classes_id2name.items()}
            for i, name in enumerate(self.dst_classes):
                key = catIdPrefix + name2id[name]
                catId2label[key] = i
                catId2name[key] = name
            # self.m_catId2label = {
            #     (catIdPrefix + cls_id): i for i, cls_id in enumerate(config.classes_id)
            # }
            # self.m_catId2name = {
            #     (catIdPrefix + cls_id): cls_name for (cls_id, cls_name) in zip(config.classes_id, self.dst_classes)
            # }
            self.m_catId2label = catId2label
            self.m_catId2name = catId2name
        else:
            self.m_catId2label = {i + bg_offset: self.dst_classes.index(cat) for i, cat in enumerate(self.dst_classes)}
            self.m_catId2name = {i + bg_offset: cat for i, cat in enumerate(self.dst_classes)}
    
    def init(self):
        ann_file = self.ann_file
        if type(ann_file) is str:
            data_infos = self.load_annotations(ann_file)
        elif isinstance(ann_file, Sequence):
            data_infos = []
            for af in ann_file:
                data_infos.extend(self.load_annotations(af))
        else:
            raise ValueError(f"Error type of ann_file: {ann_file}")
        # data_infos = self.load_annotations(ann_dir, self.ann_cache_version)
        self.m_data_infos = data_infos # self.filter_data_info(data_infos)
        val_inds = self._filter_imgs()
        self.m_data_infos = np.array(self.m_data_infos)[val_inds].tolist()
        self.stats()

    def _load_train_phase_anno(self, img_ids):
        data_infos = []
        for img_id in img_ids:
            data_info = {}

            filename = osp.join(self.m_img_dir, self.img_subdir, f'{img_id}.bmp')
            data_info['filename'] = f'{img_id}.bmp'
            xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)

            if width is None or height is None:
                img_path = osp.join(self.m_img_dir, filename)
                img = Image.open(img_path)
                width, height = img.size
            ### bsf.c 用于检测加载的文件是否正确
            data_info['id'] = img_id
            data_info['width'] = width
            data_info['height'] = height
            data_info['ann'] = self.load_ann_info(img_id, root=root)
            data_infos.append(data_info)
        return data_infos
    
    def load_ann_info(self, img_id, **kwargs):
        root = kwargs.get("root", None)
        if root is None:
            xml_path = osp.join(self.m_img_dir, self.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_headers = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []
        gt_headers_ignore = []
        ann = {}
        ids = []
        for obj in root.findall('HRSC_Objects/HRSC_Object'):
            if self.classwise:
                class_id = obj.find('Class_ID').text
                label = self.m_catId2label.get(class_id)
                if label is None:
                    continue
            else:
                label = 0
            id_node = obj.find("Object_ID")
            if id_node is not None:
                ids.append(int(id_node.text))
            else:
                ids.append(0)
            # Add an extra score to use obb2poly_np
            bbox = [[
                float(obj.find('mbox_cx').text),
                float(obj.find('mbox_cy').text),
                float(obj.find('mbox_w').text),
                float(obj.find('mbox_h').text),
                float(obj.find('mbox_ang').text), 
                0,
            ]]
            bbox = np.array(bbox, dtype=np.float32)

            polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32)
            if self.version != 'le90':
                bbox = np.array(
                    poly2obb_np(polygon, self.version), dtype=np.float32)
            else:
                bbox = bbox[0, :-1]
            header = [
                int(obj.find('header_x').text),
                int(obj.find('header_y').text)
            ]
            head = np.array(header, dtype=np.int64)

            ignore_node = obj.find("ignore")
            ignore = ignore_node is not None and int(ignore_node.text) == 1
            if ignore:
                gt_bboxes_ignore.append(bbox)
                gt_labels_ignore.append(label)
                gt_polygons_ignore.append(polygon)
                gt_headers_ignore.append(head)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_polygons.append(polygon)
                gt_headers.append(head)

        if gt_bboxes:
            ann['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            ann['labels'] = np.array(gt_labels, dtype=np.int64)
            ann['polygons'] = np.array(gt_polygons, dtype=np.float32)
            ann['headers'] = np.array(gt_headers, dtype=np.int64)
        else:
            ann['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels'] = np.array([], dtype=np.int64)
            ann['polygons'] = np.zeros((0, 8), dtype=np.float32)
            ann['headers'] = np.zeros((0, 2), dtype=np.float32)

        if gt_polygons_ignore:
            ann['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(gt_labels_ignore, dtype=np.int64)
            ann['polygons_ignore'] = np.array(gt_polygons_ignore, dtype=np.float32)
            ann['headers_ignore'] = np.array(gt_headers_ignore, dtype=np.float32)
        else:
            ann['bboxes_ignore'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels_ignore'] = np.array([], dtype=np.int64)
            ann['polygons_ignore'] = np.zeros((0, 8), dtype=np.float32)
            ann['headers_ignore'] = np.zeros((0, 2), dtype=np.float32)
        if len(ids):
            ids = np.array(ids)
            ann['ids'] = ids
        else:
            ann['ids'] = np.array([0])
        return ann

    def load_annotations(self, ann_file, cache_version=1):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of Imageset file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        # ann_loc = osp.join(self.m_img_dir, self.ann_subdir, ann_file)
        img_ids = []
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                img_ids.append(line)
        # img_ids = list(sorted(os.listdir(ann_file)))
        # img_ids = [osp.splitext(x)[0] for x in img_ids]
        data_infos = self._load_train_phase_anno(img_ids)
        
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.m_data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds
    
    def pre_pipeline(self, results, ip_idx=0):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = osp.join(self.m_img_dir, self.img_subdir)
        # results['seg_prefix'] = self.seg_prefix
        # results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields']  = []
        # results['transform'] = {}
        # results['addtion']     = ip_idx >= 1


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
        label2catId = {v: k for k, v in self.m_catId2label.items()}
        label2name = {v: self.m_catId2name[k] for k, v in self.m_catId2label.items()}
        for i in range(len(self.m_data_infos)):
            ann_info = self.get_ann_info(i)
            labels = ann_info['labels']
            if labels is None:
                no_label_count += 1
                continue
            for l in labels:
                name = label2name[l]
                counter[name] += 1
        ### sort by label
        sort_counter = {}
        filter_sort_counter = {}
        for c in self.dst_classes:
            sort_counter[c] = counter[c]
            if counter[c] > 0:
                filter_sort_counter[c] = counter[c]


        logger.info(f"Dataset objects dist ({len(sort_counter)}) for {self.ann_file}: \n{sort_counter}")
        logger.info(f"Filtered Dataset objects dist ({len(filter_sort_counter)}): {filter_sort_counter}")
        logger.info(f"CLASS CatId2Names: {self.m_catId2name}")
        logger.info(f"Total images: {len(self.m_data_infos)} Empty labels: {no_label_count}, AngleVersion: {self.version}")

