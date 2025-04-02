import os.path as osp
import numpy as np
from collections import Counter
import torch
import os
import tempfile
import copy
import logging
from PIL import Image
from pycocotools import cocoeval
from pycocotools import coco
from pycocotools import mask as mask_api
import six
import time, copy, datetime
from terminaltables import AsciiTable

from gdet.structures.datasets import ImageInfo, AnnInfo, DataBatchItems, DataTransedItems
from gdet.evaluation import eval_recalls
from gdet.registries import EVALUATORS
from gdet.core.bbox import obb2poly_np, poly2obb_np
from ..dataset.custom_dataset import CustomDataset
from .basic_evaluator import BasicEvaluator

from gdet.utils import box_utils
from gdet.utils import mask_utils

from gdet.structures.evaluation import CocoPredAnn, ModelPredictions

logger = logging.getLogger("gdet.eval")

class COCOFormater(object):    
    @staticmethod
    def xyxy2xywh(bbox: "np.ndarray"):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
    
    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.m_img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.m_img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.m_img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results
       
    
    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            with open(result_files['bbox'], "wb") as f:
                torch.save(json_results, f)
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            with open(result_files['bbox'], "wb") as f:
                torch.save(json_results[0], f)
            with open(result_files['segm'], "wb") as f:
                torch.save(json_results[1], f)                
            
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            with open(result_files['proposal'], "wb") as f:
                torch.save(json_results, f)
            
        else:
            raise TypeError('invalid type of results')
        return result_files


    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir



class COCOWrapper(coco.COCO):
    """COCO wrapper class.

    This class wraps COCO API object, which provides the following additional
    functionalities:
      1. Support string type image id.
      2. Support loading the groundtruth dataset using the external annotation
         dictionary.
      3. Support loading the prediction results using the external annotation
         dictionary.
    """

    def __init__(self, eval_type="box", annotation_file=None, gt_dataset=None):
        """Instantiates a COCO-style API object.

        Args:
          eval_type: either 'box' or 'mask'.
          annotation_file: a JSON file that stores annotations of the eval dataset.
            This is required if `gt_dataset` is not provided.
          gt_dataset: the groundtruth eval datatset in COCO API format.
        """
        if (annotation_file and gt_dataset) or (
            (not annotation_file) and (not gt_dataset)
        ):
            raise ValueError(
                "One and only one of `annotation_file` and `gt_dataset` "
                "needs to be specified."
            )

        if eval_type not in ["box", "mask"]:
            raise ValueError("The `eval_type` can only be either `box` or `mask`.")

        coco.COCO.__init__(self, annotation_file=annotation_file)
        self._eval_type = eval_type
        if gt_dataset:
            self.dataset = gt_dataset
            self.createIndex()

    def loadRes(self, predictions: "list[dict]"):
        """Loads result file and return a result api object.

        Args:
          predictions: a list of dictionary each representing an annotation in COCO
            format. The required fields are `image_id`, `category_id`, `score`,
            `bbox`, `segmentation`.

        Returns:
          res: result COCO api object.

        Raises:
          ValueError: if the set of image id from predctions is not the subset of
            the set of image id of the groundtruth dataset.
        """
        res = coco.COCO()
        res.dataset["images"] = copy.deepcopy(self.dataset["images"])
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])

        image_ids = [ann["image_id"] for ann in predictions]
        if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
            raise ValueError("Results do not correspond to the current dataset!")
        for ann in predictions:
            x1, x2, y1, y2 = [
                ann["bbox"][0],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1],
                ann["bbox"][1] + ann["bbox"][3],
            ]
            if self._eval_type == "box":
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            elif self._eval_type == "mask":
                ann["area"] = mask_api.area(ann["segmentation"])

        res.dataset["annotations"] = copy.deepcopy(predictions)
        res.createIndex()
        return res



class COCOevalWrapper(cocoeval.COCOeval):
    def __init__(self, cocoGt = None, cocoDt = None, iouType = "segm"):
        super().__init__(cocoGt, cocoDt, iouType)
        self.params.maxDets = [10, 100, 1000]
        # self.params.useCats = False
        
    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        logger.info('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            logger.info('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        logger.info('Evaluate annotation type *{}*'.format(p.iouType))
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
        self.ious = {}
        for imgId in p.imgIds:
            for catId in catIds:
                self.ious[(imgId, catId)] = computeIoU(imgId, catId)

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = []
        for catId in catIds:
            for areaRng in p.areaRng:
                for imgId in p.imgIds:
                    r = evaluateImg(imgId, catId, areaRng, maxDet)
                    self.evalImgs.append(r)
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        logger.info('DONE (t={:0.2f}s).'.format(toc-tic))

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
            """
            """
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
            # _metric_names = ["AP", "AP50", "AP75", 
            #     # "APs", # "APm", # "APl", # "ARmax1", # "ARmax10", # "ARmax100",
            #     # "ARs", # "ARm", # "ARl",
            # ]

            stats = {}
            # stats = np.zeros((12,))
            stats["AP"] = _summarize(1)
            maxDets=self.params.maxDets[-1]
            _, cat_ap_stats = _summarize(1, maxDets=maxDets, return_cat=True)
            stats["AP50"], cat_ap50_stats = _summarize(1, iouThr=.5, maxDets=maxDets, return_cat=True)
            # cat_ap50_stats = cat_stats
            ### bsf.c remove useless stats info
            maxDets=self.params.maxDets[-1]
            stats["AP75"] = _summarize(1, iouThr=.75, maxDets=maxDets)
            # stats[3] = _summarize(1, areaRng='small', maxDets=maxDets)
            # stats[4] = _summarize(1, areaRng='medium', maxDets=maxDets)
            # stats[5] = _summarize(1, areaRng='large', maxDets=maxDets)
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            maxDets=self.params.maxDets[-1]
            stats[f"ARmax{maxDets}"], cat_stats = _summarize(0, maxDets=maxDets, return_cat=True)
            category_ar_stats = cat_stats
            # stats[9] = _summarize(0, areaRng='small', maxDets=maxDets)
            # stats[10] = _summarize(0, areaRng='medium', maxDets=maxDets)
            # stats[11] = _summarize(0, areaRng='large', maxDets=maxDets)
            stats_value = np.array(list(stats.values()))
            stats['AP50'] = stats["AP50"]
            stats['cat_AP'] = cat_ap_stats
            stats['cat_AP50'] = cat_ap50_stats
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
        


@EVALUATORS.register_module()    
class CocoEvaluator(BasicEvaluator):
    coco_eval_cls = COCOevalWrapper
    def __init__(self, config, dataset: "CustomDataset"):
        super().__init__()
        self.m_dataset = dataset
        self.ann_file = config.ann_file
        self._include_mask = config.get("include_mask", False)
        self.remove_invalid_boxes = config.get("remove_invalid_boxes", False)
        self._per_category_metrics = config.get("per_category_metrics", False)
        self._metric_names = [
            "AP",
            "AP50",
            # "AP75",
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
        self._coco_gt = COCOWrapper(
            eval_type=("mask" if self._include_mask else "box"),
            annotation_file=self.ann_file,
        )
        
    def prepare_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.m_img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.m_img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def task_update(self, output):
        pass

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
        for img_name, class_preds in predictions.items():
            try:
                img_id = osp.splitext(osp.basename(img_name))[0]
                img_id = int(img_id)
            except Exception as e:
                img_id = img_name2id[img_name]
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
                    ann["bbox"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    # ann["bbox"] = obb
                    ann["segmentation"] = [poly]
                    ann["score"] = preds[i][5]
                    coco_predictions.append(ann)

        for i, ann in enumerate(coco_predictions):
            ann["id"] = i + 1

        return coco_predictions
    
    @staticmethod
    def convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
        """Converts groundtruths to the dataset in COCO format.

        Args:
        groundtruths: a dictionary of numpy arrays including the fields below.
            Note that each element in the list represent the number for a single
            example without batch dimension. K below denotes the actual number of
            instances for each image.
            Required fields:
            - source_id: a list of numpy arrays of int or string of shape
                [batch_size].
            - height: a list of numpy arrays of int of shape [batch_size].
            - width: a list of numpy arrays of int of shape [batch_size].
            - num_detections: a list of numpy arrays of int of shape [batch_size].
            - boxes: a list of numpy arrays of float of shape [batch_size, K, 4],
                where coordinates are in the original image space (not the
                normalized coordinates).
            - classes: a list of numpy arrays of int of shape [batch_size, K].
            Optional fields:
            - is_crowds: a list of numpy arrays of int of shape [batch_size, K]. If
                th field is absent, it is assumed that this instance is not crowd.
            - areas: a list of numy arrays of float of shape [batch_size, K]. If the
                field is absent, the area is calculated using either boxes or
                masks depending on which one is available.
            - masks: a list of numpy arrays of string of shape [batch_size, K],
        label_map: (optional) a dictionary that defines items from the category id
            to the category name. If `None`, collect the category mappping from the
            `groundtruths`.

        Returns:
        coco_groundtruths: the groundtruth dataset in COCO format.
        """
        source_ids = np.concatenate(groundtruths["source_id"], axis=0)
        heights = np.concatenate(groundtruths["height"], axis=0)
        widths = np.concatenate(groundtruths["width"], axis=0)
        gt_images = [
            {"id": int(i), "height": int(h), "width": int(w)}
            for i, h, w in zip(source_ids, heights, widths)
        ]

        gt_annotations = []
        num_batches = len(groundtruths["source_id"])
        for i in range(num_batches):
            # NOTE: Batch size may differ between chunks.
            batch_size = groundtruths["source_id"][i].shape[0]
            max_num_instances = groundtruths["classes"][i].shape[1]
            for j in range(batch_size):
                num_instances = int(groundtruths["num_detections"][i][j])
                if num_instances > max_num_instances:
                    logger.warning(
                        "num_groundtruths is larger than max_num_instances, %d v.s. %d",
                        num_instances,
                        max_num_instances,
                    )
                    num_instances = max_num_instances
                for k in range(num_instances):
                    ann = {}
                    ann["image_id"] = int(groundtruths["source_id"][i][j])
                    if "is_crowds" in groundtruths:
                        ann["iscrowd"] = int(groundtruths["is_crowds"][i][j, k])
                    else:
                        ann["iscrowd"] = 0
                    ann["category_id"] = int(groundtruths["classes"][i][j, k])
                    boxes = groundtruths["boxes"][i]
                    ann["bbox"] = [
                        float(boxes[j, k, 1]),
                        float(boxes[j, k, 0]),
                        float(boxes[j, k, 3] - boxes[j, k, 1]),
                        float(boxes[j, k, 2] - boxes[j, k, 0]),
                    ]
                    if "areas" in groundtruths:
                        ann["area"] = float(groundtruths["areas"][i][j, k])
                    else:
                        ann["area"] = float(
                            (boxes[j, k, 3] - boxes[j, k, 1])
                            * (boxes[j, k, 2] - boxes[j, k, 0])
                        )
                    if "masks" in groundtruths:
                        mask = Image.open(six.BytesIO(groundtruths["masks"][i][j, k]))
                        np_mask = np.array(mask, dtype=np.uint8)
                        np_mask[np_mask > 0] = 255
                        encoded_mask = mask_api.encode(np.asfortranarray(np_mask))
                        ann["segmentation"] = encoded_mask
                        if "areas" not in groundtruths:
                            ann["area"] = mask_api.area(encoded_mask)
                    gt_annotations.append(ann)

        for i, ann in enumerate(gt_annotations):
            ann["id"] = i + 1

        if label_map:
            gt_categories = [{"id": i, "name": label_map[i]} for i in label_map]
        else:
            category_ids = [gt["category_id"] for gt in gt_annotations]
            gt_categories = [{"id": i} for i in set(category_ids)]

        gt_dataset = {
            "images": gt_images,
            "categories": gt_categories,
            "annotations": copy.deepcopy(gt_annotations),
        }
        return gt_dataset


    def evaluate(self,
                 results: "dict",
                 metric='mAP',
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4, **kwargs):
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
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']

        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.m_dataset.get_ann_info(i) for i in range(len(self.m_dataset))]
        filenames = [osp.basename(di['filename']) for di in self.m_dataset.m_data_infos]
        assert len(annotations) == len(results) and list(results.keys()) == filenames
        result_list = list(results.values())
        
        coco_gt = self._coco_gt
        coco_predictions = self.convert_predictions_to_coco_annotations(
            results, self.m_dataset, self.m_dataset.version,
            remove_invalid_boxes=self.remove_invalid_boxes,
        )
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)
        image_ids = [ann["image_id"] for ann in coco_predictions]

        coco_eval = self.coco_eval_cls(coco_gt, coco_dt, iouType="bbox")
        coco_eval.version = self.m_dataset.version
        coco_eval.params.imgIds = image_ids
        # coco_eval.params.iouThrs = np.array([0.5,])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        coco_metrics_dict = coco_eval.stats_dict

        if self._include_mask:  ### bsf.TODO
            mcoco_eval = self.coco_eval_cls(coco_gt, coco_dt, iouType="segm")
            mcoco_eval.params.imgIds = image_ids
            mcoco_eval.evaluate()
            mcoco_eval.accumulate()
            mcoco_eval.summarize()
            mask_coco_metrics = mcoco_eval.stats
            coco_metrics_dict = mcoco_eval.stats_dict
        else:
            pass

        # Cleans up the internal variables in order for a fresh eval next time.
        self.reset()
        # # bsf.c Adds metrics per category.
        per_cat_metric = None
        if self._per_category_metrics:
            per_cat_metric = self.calc_per_category_metric(coco_metrics_dict, coco_eval)
        
        base_cat_len = len(self.m_dataset.base_classes)
        cat_ap = coco_metrics_dict['cat_AP']
        logger.info(f"AP50: {np.mean(cat_ap):.3f}")
        logger.info(f"bAP50: {np.mean(cat_ap[:base_cat_len]):.3f}")
        logger.info(f"nAP50: {np.mean(cat_ap[base_cat_len:]):.3f}")
        # print(coco_metrics_dict)
        cat_ar50 = coco_metrics_dict['cat_ARmax100']
        header = ['idx', 'class', 'recall', 'ap']
        table_data = [header]
        num_classes = len(self.m_dataset.m_catId2name)
        if coco_eval.params.useCats:        
            for ci, j in enumerate(range(num_classes)):
                row_data = [
                    ci, self.m_dataset.dst_classes[ci], f'{cat_ar50[j]:.3f}', f'{cat_ap[j]:.3f}'
                ]
                table_data.append(row_data)
        table_data.append(['mAP', '', '', f'{np.mean(cat_ap):.3f}'])
        table_data.append(['bAP', '', '', f'{np.mean(cat_ap[:base_cat_len]):.3f}'])
        table_data.append(['nAP', '', '', f'{np.mean(cat_ap[base_cat_len:]):.3f}'])
        table = AsciiTable(table_data)
        target_row_index = - 3

        # 创建一个分隔符行（与表格宽度一致）
        separator = ['─' * len(cell) for cell in table_data[0]]  # 根据每列的宽度生成分隔符
        # 在倒数第三行的下方插入分隔符
        table_data.insert(target_row_index, separator)
        # table.inner_footing_row_border = True
        logger.info('\n' + table.table)
        return coco_metrics_dict
    

        eval_results = {}
        dst_classes = self.m_dataset.dst_classes
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                result_list,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=dst_classes,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results
    

