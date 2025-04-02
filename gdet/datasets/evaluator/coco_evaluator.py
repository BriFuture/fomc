# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The COCO-style evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = COCOEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update(predictions, groundtruths)  # aggregate internal stats.
    evaluator.evaluate()  # finish one full eval.

See also: https://github.com/cocodataset/cocoapi/
"""

import json

import logging
import numpy as np
from pycocotools import cocoeval
import itertools
import torch
from terminaltables import AsciiTable
from gdet.registries import EVALUATORS
from . import coco_utils
logger = logging.getLogger("gdet.eval")

@EVALUATORS.register_module()
class COCOEvaluator():
    def __init__(
        self,
        annotation_file,
        include_mask = False,
        need_rescale_bboxes=True,
        per_category_metrics=False,
        remove_invalid_boxes=False,
    ):
        """Constructs COCO evaluation class.

        The class provides the interface to metrics_fn in TPUEstimator. The
        _update_op() takes detections from each image and push them to
        self.detections. The _evaluate() loads a JSON file in COCO annotation format
        as the groundtruths and runs COCO evaluation.


        There are two ways to remove invalid boxes when evaluating on COCO: 1. Set
        the coordinates of invalid boxes to "0"; 2. Set "remove_invalid_boxes" to be
        true. To make this function backward compatible, we set the default value of
        "remove_invalid_boxes" to be false.

        Args:
          annotation_file: a JSON file that stores annotations of the eval dataset.
            If `annotation_file` is None, groundtruth annotations will be loaded
            from the dataloader.
          include_mask: a boolean to indicate whether or not to include the mask
            eval.
          need_rescale_bboxes: If true bboxes in `predictions` will be rescaled back
            to absolute values (`image_info` is needed in this case).
          per_category_metrics: Whether to return per category metrics.
          remove_invalid_boxes: A boolean indicating whether to remove invalid box
            during evaluation.
        """
        if annotation_file:
            local_val_json = annotation_file
            self._coco_gt = coco_utils.COCOWrapper(
                eval_type=("mask" if include_mask else "box"),
                annotation_file=local_val_json,
            )
        self._annotation_file = annotation_file
        self._include_mask = include_mask
        self._per_category_metrics = per_category_metrics
        ## bsf.c 移除不需要的 metric
        self._metric_names = [
            "AP",
            "AP50",
            "AP75",
            # "APs",
            # "APm",
            # "APl",
            # "ARmax1",
            # "ARmax10",
            # "ARmax100",
            # "ARs",
            # "ARm",
            # "ARl",
        ]
        self._required_prediction_fields = [
            "source_id",
            "num_detections",
            "detection_classes",
            "detection_scores",
            "detection_boxes",
        ]
        self._need_rescale_bboxes = need_rescale_bboxes
        if self._need_rescale_bboxes:
            self._required_prediction_fields.append("image_info")
        self._required_groundtruth_fields = [
            "source_id",
            "height",
            "width",
            "classes",
            "boxes",
        ]
        if self._include_mask:
            mask_metric_names = ["mask_" + x for x in self._metric_names]
            self._metric_names.extend(mask_metric_names)
            self._required_prediction_fields.extend(["detection_masks"])
            self._required_groundtruth_fields.extend(["masks"])
        self.remove_invalid_boxes = remove_invalid_boxes
        self.reset()

    def evaluate(self, results, metric='bbox', logger=None,
                 jsonfile_prefix=None, classwise=False,
                 proposal_nums=(100, 300, 1000), iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logger.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            logger.info(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                logger.info(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logger.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.m_cat_ids
            cocoEval.params.imgIds = self.m_img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    
    def reset(self):
        """Resets internal states for a fresh run."""
        self._predictions = {}
        if not self._annotation_file:
            self._groundtruths = {}

    def dump_predictions(self, file_path: "str"):
        """Dumps the predictions in COCO format.

        This can be used to output the prediction results in COCO format, for
        example to prepare for test-dev result submission.

        Args:
          file_path: a string specifying the path to the prediction JSON file.
        """
        coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
            self._predictions
        )

        with open(file_path, "w") as f:
            json.dump(coco_predictions, f)

    def evaluate(self):
        """Evaluates with detections from all images with COCO API.

        Returns:
          coco_metric: float numpy array with shape [24] representing the
            coco-style evaluation metrics (box and mask).
        """
        if not self._annotation_file:
            logger.info("There is no annotation_file in COCOEvaluator.")
            gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(
                self._groundtruths
            )
            coco_gt = coco_utils.COCOWrapper(
                eval_type=("mask" if self._include_mask else "box"),
                gt_dataset=gt_dataset,
            )
        else:
            logger.info("Using annotation file: %s", self._annotation_file)
            coco_gt = self._coco_gt
        coco_predictions = coco_utils.convert_predictions_to_coco_annotations(
            self._predictions,
            remove_invalid_boxes=self.remove_invalid_boxes,
        )
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)
        image_ids = [ann["image_id"] for ann in coco_predictions]

        coco_eval = COCOevalWrapper(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.params.iouThrs = np.array([0.5, 0.75])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        coco_metrics_dict = coco_eval.stats_dict

        if self._include_mask:  ### bsf.TODO
            mcoco_eval = COCOevalWrapper(coco_gt, coco_dt, iouType="segm")
            mcoco_eval.params.imgIds = image_ids
            mcoco_eval.evaluate()
            mcoco_eval.accumulate()
            mcoco_eval.summarize()
            mask_coco_metrics = mcoco_eval.stats
            metrics = np.hstack((coco_metrics, mask_coco_metrics))
        else:
            metrics = coco_metrics

        # Cleans up the internal variables in order for a fresh eval next time.
        self.reset()
        # # bsf.c Adds metrics per category.
        per_cat_metrci = None
        if self._per_category_metrics:
            per_cat_metrci = self.calc_per_category_metric(coco_metrics_dict, coco_eval)
        return coco_metrics_dict, per_cat_metrci
    
    def calc_per_category_metric(self, coco_metrics_dict, coco_eval: "COCOevalWrapper"):
        metrics_dict = {
            "AP50": coco_metrics_dict["AP50"]
        }

        cecs = coco_metrics_dict["cat_AP50"]
        cecs_ar = coco_metrics_dict["cat_ARmax100"]
        for category_index, category_id in enumerate(coco_eval.params.catIds):
            # key = "Precision mAP ByCategory/{}".format(category_id)
            # metrics_dict[key] = cecs[0][category_index].astype(np.float32)
            key = "Precision mAP ByCategory@50IoU/{}".format(category_id)
            metrics_dict[key] = cecs[category_index]
        for category_index, category_id in enumerate(coco_eval.params.catIds):
            key = "Precision AR@100 ByCategory/{}".format(category_id)
            metrics_dict[key] = cecs_ar[category_index]
        return metrics_dict

    def _process_predictions(self, predictions):
        image_scale = np.tile(predictions["image_info"][:, 2:3, :], (1, 1, 2))
        predictions["detection_boxes"] = predictions["detection_boxes"].astype(
            np.float32
        )
        predictions["detection_boxes"] /= image_scale
        if "detection_outer_boxes" in predictions:
            predictions["detection_outer_boxes"] = predictions[
                "detection_outer_boxes"
            ].astype(np.float32)
            predictions["detection_outer_boxes"] /= image_scale

    def update(self, predictions: "dict", groundtruths: "dict"=None):
        """Update and aggregate detection results and groundtruth data.

        Args:
          predictions: a dictionary of numpy arrays including the fields below.
            See different parsers under `../dataloader` for more details.
            Required fields:
              - source_id: a numpy array of int or string of shape [batch_size].
              - image_info [if `need_rescale_bboxes` is True]: a numpy array of
                float of shape [batch_size, 4, 2].
              - num_detections: a numpy array of
                int of shape [batch_size].
              - detection_boxes: a numpy array of float of shape [batch_size, K, 4].
              - detection_classes: a numpy array of int of shape [batch_size, K].
              - detection_scores: a numpy array of float of shape [batch_size, K].
            Optional fields:
              - detection_masks: a numpy array of float of shape
                  [batch_size, K, mask_height, mask_width].
          groundtruths: a dictionary of numpy arrays including the fields below.
            See also different parsers under `../dataloader` for more details.
            Required fields:
              - source_id: a numpy array of int or string of shape [batch_size].
              - height: a numpy array of int of shape [batch_size].
              - width: a numpy array of int of shape [batch_size].
              - num_detections: a numpy array of int of shape [batch_size].
              - boxes: a numpy array of float of shape [batch_size, K, 4].
              - classes: a numpy array of int of shape [batch_size, K].
            Optional fields:
              - is_crowds: a numpy array of int of shape [batch_size, K]. If the
                  field is absent, it is assumed that this instance is not crowd.
              - areas: a numy array of float of shape [batch_size, K]. If the
                  field is absent, the area is calculated using either boxes or
                  masks depending on which one is available.
              - masks: a numpy array of float of shape
                  [batch_size, K, mask_height, mask_width],

        Raises:
          ValueError: if the required prediction or groundtruth fields are not
            present in the incoming `predictions` or `groundtruths`.
        """
        for k in self._required_prediction_fields:
            if k not in predictions:
                raise ValueError(
                    "Missing the required key `{}` in predictions!".format(k)
                )
        if self._need_rescale_bboxes: ## 
            self._process_predictions(predictions)
        for k, v in predictions.items():
            if k not in self._predictions:
                self._predictions[k] = [v]
            else:
                self._predictions[k].append(v)

        if not self._annotation_file:
            assert groundtruths
            for k in self._required_groundtruth_fields:
                if k not in groundtruths:
                    raise ValueError(
                        "Missing the required key `{}` in groundtruths!".format(k)
                    )
            for k, v in groundtruths:
                if k not in self._groundtruths:
                    self._groundtruths[k] = [v]
                else:
                    self._groundtruths[k].append(v)


import time, copy, datetime
class COCOevalWrapper(cocoeval.COCOeval):

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, return_cat=False ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if return_cat:
                ## bsf.c get category metrics
                catIds = p.catIds if p.useCats == 1 else [-1]
                cat_mean_s = []
                if ap:
                    for i, c in enumerate(catIds):
                        cs = s[:, :, i, ...]
                        mask = cs > -1
                        cat_mean_s.append(np.mean(cs[mask]) if mask.any() else np.float32(0))
                else:
                    for i, c in enumerate(catIds):
                        cs = s[:, i, ...]
                        mask = cs > -1
                        cat_mean_s.append(np.mean(cs[mask]) if mask.any() else np.float32(0))
                logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s, cat_mean_s
            else:
                logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s
        def _summarizeDets():
            ### bsf.c check stats as dict
            _metric_names = ["AP", "AP50", "AP75", 
                # "APs", # "APm", # "APl", # "ARmax1", # "ARmax10", # "ARmax100",
                # "ARs", # "ARm", # "ARl",
            ]
            stats = {}
            # stats = np.zeros((12,))
            stats["AP"] = _summarize(1)
            stats["AP50"], cat_stats = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], return_cat=True)
            category_stats = cat_stats
            ### bsf.c remove useless stats info
            maxDets=self.params.maxDets[2]
            stats["AP75"] = _summarize(1, iouThr=.75, maxDets=maxDets)
            # stats[3] = _summarize(1, areaRng='small', maxDets=maxDets)
            # stats[4] = _summarize(1, areaRng='medium', maxDets=maxDets)
            # stats[5] = _summarize(1, areaRng='large', maxDets=maxDets)
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            maxDets=self.params.maxDets[2]
            stats[f"ARmax{maxDets}"], cat_stats = _summarize(0, maxDets=maxDets, return_cat=True)
            category_ar_stats = cat_stats
            # stats[9] = _summarize(0, areaRng='small', maxDets=maxDets)
            # stats[10] = _summarize(0, areaRng='medium', maxDets=maxDets)
            # stats[11] = _summarize(0, areaRng='large', maxDets=maxDets)
            stats_value = np.array(list(stats.values()))
            stats['AP50'] = stats["AP50"]
            stats['cat_AP50'] = category_stats
            stats['cat_ARmax100'] = category_ar_stats
            return stats_value, stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            self.stats, self.stats_dict = _summarizeDets()
        elif iouType == 'keypoints':
            self.stats = _summarizeKps()
        