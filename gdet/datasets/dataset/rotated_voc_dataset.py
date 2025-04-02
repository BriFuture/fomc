import os, os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import logging

from .custom_dataset import CustomDataset
from .voc_dataset import VocDataset
from gdet.registries import DATASETS
from mmrotate.core import poly2obb_np, obb2poly
logger = logging.getLogger("gdet.datasets.dota")

@DATASETS.register_module()
class RotateVocDataset(VocDataset):
    def __init__(self, config, min_size=None, **kwargs):
        super().__init__(config, min_size, **kwargs)
        self.version = config.version

    def load_ann_info(self, img_id, **kwargs):
        """Get annotation from XML file by index.

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
        gt_polygons = []
        ids_ignore = []
        bboxes_ignore = []
        labels_ignore = []
        gt_polygons_ignore = []
        name2catid = {v: k for k, v in self.m_catId2name.items()}
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.dst_classes: # or name != 'airplane':
                continue
            label = name2catid[name]
            diff_node = obj.find('difficult')
            difficult = int(diff_node.text) if diff_node else 0
            
            id_node = obj.find('oid')
            
            bnd_box = obj.find('robndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            poly = [
                float(bnd_box.find('x_left_top').text),
                float(bnd_box.find('y_left_top').text),
                float(bnd_box.find('x_right_top').text),
                float(bnd_box.find('y_right_top').text),
                float(bnd_box.find('x_right_bottom').text),
                float(bnd_box.find('y_right_bottom').text),
                float(bnd_box.find('x_left_bottom').text),
                float(bnd_box.find('y_left_bottom').text),
            ]
            try:
                data = poly2obb_np(poly, self.version)
            except Exception as e:  # noqa: E722
                print("Err loading rvoc dataset", e)
                continue
            if data is None:
                continue
            x, y, w, h, a = data
            bbox = [x, y, w, h, a]
           
            ignore_element = obj.find("ignore")
            if ignore_element is not None:
                ignore = int(ignore_element.text) > 0
            else:
                ignore = False
            if self.min_size:
                assert not self.test_mode
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
                gt_polygons_ignore.append(poly)
                if id_node is not None:
                    ids_ignore.append(int(id_node.text))
                else:
                    ids_ignore.append(0)
            else:
                bboxes.append(bbox)
                labels.append(label)
                gt_polygons.append(poly)
                if id_node is not None:
                    ids.append(int(id_node.text))
                else:
                    ids.append(0)

        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0, ))
            ids = np.zeros((0, ))
            polygons = np.zeros((0, 8))
        else:
            bboxes = np.array(bboxes, ndmin=2) 
            labels = np.array(labels)
            ids = np.array(ids)
            polygons = np.array(gt_polygons, dtype=np.float32)
        ann = dict(
            ids=ids,
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            polygons= polygons.astype(np.float32),
        )
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0, ))
            gt_polygons_ignore = np.zeros((0, 8))
            ids_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) 
            labels_ignore = np.array(labels_ignore)
            ids_ignore = np.array(ids_ignore)
            gt_polygons_ignore = np.array(gt_polygons_ignore, ndmin=2) 

        ann['polygons_ignore'] = gt_polygons_ignore.astype(np.float32)
        ann['bboxes_ignore'] = bboxes_ignore.astype(np.float32)
        ann['labels_ignore'] = labels_ignore.astype(np.float32)
        ann['ids_ignore'] = ids_ignore.astype(np.float32)
            
        return ann
    
