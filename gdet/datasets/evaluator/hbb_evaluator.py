import os, time, re
import numpy as np
import logging
from ..dataset.custom_dataset import CustomDataset
from gdet.registries import EVALUATORS
from mmdet.core.evaluation.mean_ap import print_map_summary, Pool, tpfp_imagenet, tpfp_openimages, tpfp_default, get_cls_group_ofs, average_precision
from gdet.datasets.evaluator.utils import tree_map, detach_tensor

from .basic_evaluator import BasicEvaluator

logger = logging.getLogger("gdet.eval.hbb")

def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[dict]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    # cls_dets = [img_res[class_id] for img_res in det_results]
    cls_dets = []
    cls_gts = []
    cls_gts_ignore = []
    for det, ann in zip(det_results, annotations):
        det_inds = det['class'] == class_id
        cls_dets.append(det['bbox'][det_inds, :])
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore

def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             ioa_thr=None,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4,
             num_classes = 0,
             use_legacy_coordinate=False,
             use_group_of=False):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: False.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    if num_classes == 0:
        num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # There is no need to use multi processes to process
    # when num_imgs = 1 .
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            elif dataset in ['oid_challenge', 'oid_v6'] \
                    or use_group_of is True:
                tpfp_fn = tpfp_openimages
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        if num_imgs > 1:
            # compute tp and fp for each image with multiple processes
            args = []
            if use_group_of:
                # used in Open Images Dataset evaluation
                gt_group_ofs = get_cls_group_ofs(annotations, i)
                args.append(gt_group_ofs)
                args.append([use_group_of for _ in range(num_imgs)])
            if ioa_thr is not None:
                args.append([ioa_thr for _ in range(num_imgs)])

            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)],
                    [use_legacy_coordinate for _ in range(num_imgs)], *args))
        else:
            tpfp = tpfp_fn(
                cls_dets[0],
                cls_gts[0],
                cls_gts_ignore[0],
                iou_thr,
                area_ranges,
                use_legacy_coordinate,
                gt_bboxes_group_of=(get_cls_group_ofs(annotations, i)[0]
                                    if use_group_of else None),
                use_group_of=use_group_of,
                ioa_thr=ioa_thr)
            tpfp = [tpfp]

        if use_group_of:
            tp, fp, cls_dets = tuple(zip(*tpfp))
        else:
            tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
                    bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    if num_imgs > 1:
        pool.close()

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


@EVALUATORS.register_module()
class HBB_Evaluator(BasicEvaluator):
    val_metric_key = "AP50"

    def __init__(self, config, dataset: "CustomDataset"):
        super().__init__()
        self.m_dataset = dataset
        
        pass

    def process_predictions(self, det):
        det = tree_map(detach_tensor, det)

        self._process_predictions(det)
        return det

    def task_update(self, output):
        detection = output['detection']
        det = self.process_predictions(detection)
        for idx in range(len(det['source_id'])):

            ids = det['source_id'][idx].item()
            num_dets = det['num_detections'][idx]
            det_boxes = det['detection_boxes'][idx]
            det_scores = det['detection_scores'][idx]
            det_classes = det['detection_classes'][idx]
            det_classes[num_dets:] = 0
            self._predictions[ids] = {
                "bbox": det_boxes, "score": det_scores, "class": det_classes, 
            }


    def task_evaluate(self, indicator):
        metric = 'mAP'
        eval_results = self.evaluate(self._predictions, metric)
        eval_results[self.val_metric_key] = eval_results[metric]
        return eval_results

    def evaluate(self,
                 results: "dict",
                 metric='mAP',
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.m_dataset.get_ann_info(i) for i in range(len(self.m_dataset))]
        filenames = [di['ann']['ids'][0] for di in self.m_dataset.m_data_infos]
        assert len(annotations) == len(results) and list(results.keys()) == filenames
        result_list = list(results.values())

        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            dst_classes = ['background'] + list(self.m_dataset.dst_classes)
            mean_ap, _ = eval_map(
                result_list,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=dst_classes,
                logger=logger,
                num_classes=len(self.m_dataset.dst_classes) + 1,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results
