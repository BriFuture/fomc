import sys
import os.path as osp
import numpy as np
from collections import Counter
import torch
import os
import tempfile
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon
import shapely
from shapely.geometry import Polygon, MultiPoint
import copy
import pycocotools.mask as maskUtils

from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from gdet.evaluation import eval_recalls
from gdet.registries import EVALUATORS
from gdet.core.bbox import obb2poly_np, poly2obb_np
from .coco import CocoEvaluator, COCOWrapper, COCOevalWrapper
from gdet.structures.evaluation import CocoPredAnn, ModelPredictions



class RotatedCOCO(COCOWrapper):
    def loadRes(self, predictions):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(predictions) == str:
            with open(predictions) as f:
                anns = json.load(f)
        elif type(predictions) == np.ndarray:
            anns = self.loadNumpyAnnotations(predictions)
        else:
            anns = predictions # list
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and anns[0]['bbox'].shape[0] != 0:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                [x1, y1, x2, y2, x3, y3, x4, y4] = bb
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x2, y2, x3, y3, x4, y4, x1, y1]]
                ann['area'] = Polygon([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]).convex_hull.area
                # ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

from gdet.core.bbox.poly_iou_rotated import poly_iou_rotated_np
class RotatedCOCOevalWrapper(COCOevalWrapper):
    version = 'oc'
    @staticmethod
    def is_rotated(box_list):
        if type(box_list) == np.ndarray:
            return box_list.shape[1] == 8
        elif type(box_list) == list:
            if box_list == []:  # cannot decide the box_dim
                return False
            return np.all(
                np.array(
                    [
                        (len(obj) == 8) and ((type(obj) == list) or (type(obj) == np.ndarray))
                        for obj in box_list
                    ]
                )
            )
        return False

    def compute_iou_dt_gt(self, dt, gt, is_crowd):
            # TODO: take is_crowd into consideration
        assert all(c == 0 for c in is_crowd)
        # iou = polygon_box_iou(torch.FloatTensor(dt), torch.FloatTensor(gt))
        iou = poly_iou_rotated_np(dt, gt, self.version)
        return iou

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"

        # g = [g["bbox"] for g in gt]
        g = [g["segmentation"][0] for g in gt]  ### bsf.c segmentation nx8
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Note: this function is copied from cocoeval.py in cocoapi
        # and the major difference is here.
        ious = self.compute_iou_dt_gt(d, g, iscrowd)
        return ious
    


@EVALUATORS.register_module()    
class RotatedCocoEvaluator(CocoEvaluator):
    coco_eval_cls = RotatedCOCOevalWrapper
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._coco_gt = RotatedCOCO(            
            eval_type=("mask" if self._include_mask else "box"),
            annotation_file=self.ann_file,
        )
    @staticmethod
    def convert_predictions_to_coco_annotations(
        predictions: "dict", dataset, version='oc', remove_invalid_boxes=False, 
    ):
        """Converts a batch of predictions to annotations in COCO format.

        Returns:
        coco_predictions: prediction in COCO annotation format.
        """
        coco_predictions: "list[CocoPredAnn]" = []
        # num_preds = len(predictions)
        # max_num_detections = predictions["detection_classes"][0].shape[1]  ### detection_classes is list[list[np.ndarray]]
        # use_outer_box = "detection_outer_boxes" in predictions
        img_name2id = {k['filename']: k['id'] for k in dataset.m_data_infos}
        # g_img_id = 1
        for img_name, class_preds in predictions.items():
            img_id = img_name2id[img_name]
                # g_img_id += 1
            # num_class = len(class_preds)
            for cls_idx, preds in enumerate(class_preds):
                cls_id = cls_idx + 1 ### coco 映射时 bg 为 0
                for i in range(preds.shape[0]):
                    ann: "CocoPredAnn" = {}
                    ann["image_id"] = img_id
                    ann["category_id"] = cls_id
                    obb = preds[i]
                    polys = obb2poly_np(np.asarray([obb]), version)
                    obb = obb[:5]
                    poly = polys[0, :8]
                    bbox = np.zeros((4,), dtype=int)
                    bbox[0] = min(poly[0::2])
                    bbox[1] = min(poly[1::2])
                    bbox[2] = max(poly[0::2])
                    bbox[3] = max(poly[1::2])
                    ann["bbox"] = poly
                    ann["segmentation"] = [poly]
                    ann["score"] = preds[i][5]
                    coco_predictions.append(ann)
        ### 设置 prediction id
        for i, ann in enumerate(coco_predictions):
            ann["id"] = i + 1

        return coco_predictions    