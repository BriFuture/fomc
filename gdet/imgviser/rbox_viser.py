import os, os.path as osp
import logging
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as maskUtils

from gdet.registries import VISER
from bfcommon.utils import convert_tensor_as_npint8
from .basevistools import BaseViser

@VISER.register_module()
class RBoxViserTrain(BaseViser):
    def plot(self, img,
                      bboxes: "list[np.ndarray]",
                      labels: "list[np.ndarray]",
                      out_file=None, vis=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 8) or
                (n, 9).
            labels (ndarray): Labels of bboxes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.
        """
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        img = cv2.imread(img)
        class_names = self.class_names
        class_colors = self.class_colors
        font_scale = self.font_scale
        # class_names, full_class_names = class_names
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)[:-1].reshape(-1, 2)
            if label > len(class_names):
                label_text = "bg"
                continue
            else:
                label_text = class_names[label]

            bbox_color = class_colors[label_text]
            text_color = bbox_color
            for i in range(4):
                cv2.line(img, bbox_int[i], bbox_int[(i+1) % 4], bbox_color, thickness=self.thickness)
            if len(bbox) > 8:
                label_text += '|{:.02f}'.format(bbox[-1] + 0.05)
                # label_text = f'{bbox[-1]:.02f}'
            tl = (bbox_int[0][0], bbox_int[0][1] - 2)
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, 3)
            text_w, text_h = text_size
            cv2.rectangle(img, (bbox_int[0][0], bbox_int[0][1] - text_h - 2), (bbox_int[0][0] + text_w, bbox_int[0][1] - 2), (255, 255, 255), -1)
            cv2.putText(img, label_text, tl,
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if vis:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            text_color = (10,10,10)
            cv2.putText(img, out_file, (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        elif out_file is not None:
            cv2.imwrite(out_file, img,)

from gdet.core.bbox import obb2poly_np, poly_iou_rotated_np, box_iou_rotated_np

def find_match_most_output_poly(plot_output, gt_boxes, version='oc'):
    """plot_output 和 gt_bboxes 需要是 poly  dim=8
    """
    match_plot_output = []
    for poutput, gt_box in zip(plot_output, gt_boxes):
        match_poutput = []
        for po in poutput:
            if po.shape[0] == 0:
                match_poutput.append(po)
                continue
            
            po = po[:, :8]
            ious = poly_iou_rotated_np(po, gt_box, version)
            idx = np.argmax(ious, axis=0)
            match_poutput.append(po[idx])
        match_plot_output.append(match_poutput)
    return match_plot_output

def find_match_most_output_obb(plot_output, gt_boxes, version='oc'):
    """plot_output 和 gt_bboxes 需要是 obb   dim=5
    """
    match_plot_output = []
    for poutput, gt_box in zip(plot_output, gt_boxes):
        match_poutput = []
        for po in poutput:
            if po.shape[0] == 0:
                match_poutput.append(po)
                continue
            ### bsf.c 这里输入的是 obb(xywha)
            # po = po
            ious = box_iou_rotated_np(po[:, :5], gt_box,)
            if (ious == 0).all():
                match_poutput.append(po)
            else:
                idx = np.argmax(ious, axis=0)
                match_poutput.append(po[idx, :])
        match_poutput = np.concatenate(match_poutput)
        match_plot_output.append(match_poutput)
    return match_plot_output

def revert_obb_pd(outputs: "list[list]", version: str):
    """将 obb (xywhas) dim=6 转为 poly (x1y1x2y2x3y3x4y4s) dim=9
    """
    n_outputs = []
    for output in outputs:
        n_output = []
        for out in output:
            poly = obb2poly_np(out, version)
            n_output.append(poly.astype(np.float32))
        n_outputs.append(n_output)
    return n_outputs

def revert_obb_gt(outputs: "list[list]", version: str):
    """将 obb (xywhas) dim=6 转为 poly (x1y1x2y2x3y3x4y4s) dim=9
    """
    n_outputs = []
    for output in outputs:
        if output.shape[1] != 6:
            zero = np.zeros((output.shape[0], 6))
            zero[:, :5] = output
            output = zero
        poly = obb2poly_np(output, version)
        n_outputs.append(poly[:, :8])
    return n_outputs

@VISER.register_module()
class RBoxViser(BaseViser):


    def init(self, img_root = None):
        if img_root is None:
            img_root = "checkpoints/analyse/gt_rbox"
        self.img_root = img_root
        os.makedirs(self.img_root, exist_ok=True)
        self.imgs = []
        self.offset = 0
        self.overwrite = True
        self.fake_box = False
        
    def set_images(self, ximgs: "torch.Tensor", scales=None):
        self.imgs = []
        for imidx, xim in enumerate(ximgs):
            if isinstance(xim, torch.Tensor):
                image, m1, m2 = convert_tensor_as_npint8(xim)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif isinstance(xim, np.ndarray):
                image = xim.astype(np.int8)
            elif type(xim) is str:
                image = cv2.imread(xim)
            if scales is not None:
                scale = scales[imidx]
                hr, wr = scale.detach().cpu().numpy()
                original_height, original_width = image.shape[:2]
                new_width = int(original_width / wr)
                new_height = int(original_height / hr)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            self.imgs.append(image)

    def cplot(self, gt_bboxes: "list[torch.Tensor]", cls_labels: "list[str]"=None, color=(0, 0, 225)):
        nimgs = []
        for imidx, image in enumerate(self.imgs):
            if self.fake_box:
                nimgs.append(image)
                continue
            bbox = gt_bboxes[imidx]
            if bbox is None:
                nimgs.append(image)
                continue
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.detach().cpu().numpy()
            # polygons = box_to_poly_np(bbox)
            polygons = bbox
            if cls_labels is not None:
                cls_label = cls_labels[imidx]
            for pi, poly in enumerate(polygons):
                if cls_labels is not None:
                    label = cls_label[pi]
                else:
                    label = ""
                poly = np.array(poly, dtype=np.int32)
                poly = poly.reshape((-1, 2))
                for i in range(3):
                    image = cv2.line(image, poly[i], poly[i+1], color, 1, lineType=cv2.LINE_AA)
                image = cv2.line(image, poly[3], poly[0], color, 3, lineType=cv2.LINE_AA)
                center = ((poly[0][0] + poly[2][0]) // 2, (poly[0][1] + poly[2][1]) // 2)
                image = cv2.putText(image, f"{pi}-{label}", center, cv2.FONT_HERSHEY_SIMPLEX, 
                                    color=(0, 0, 0), fontScale=1)
            nimgs.append(image)
        self.imgs = nimgs

    def cplot_with_label(self, gt_bboxes: "list[torch.Tensor]", cls_labels: "list[str]",  color=(0, 0, 225), score_thres=0.05):
        """"gt_bboxes list[list[x1, y1, x2, y2, x3, y3, x4, y4, score]]
        gt_bboxes list[list[x, y, w, h, a, score]]
        """
        nimgs = []
        for imidx, image in enumerate(self.imgs):
            if self.fake_box:
                nimgs.append(image)
                continue
            gt_bbox = gt_bboxes[imidx]
            for cls_id, bbox  in enumerate(gt_bbox):
                label = cls_labels[cls_id]
                ### bsf.c bbox 可能为空，为空时画出原图
                if bbox.shape[0] == 0:
                    continue
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.detach().cpu().numpy()
                # polygons = box_to_poly_np(bbox)
                scores = bbox[:, -1]
                mask = scores > score_thres
                polygons = bbox[mask]
                for pi, poly in enumerate(polygons):
                    poly = np.array(poly, dtype=np.int32)
                    poly = poly.reshape((-1, 2))
                    for i in range(3):
                        image = cv2.line(image, poly[i], poly[i+1], color, 1, lineType=cv2.LINE_AA)
                    image = cv2.line(image, poly[3], poly[0], color, 3, lineType=cv2.LINE_AA)
                    center = ((poly[0][0] + poly[2][0]) // 2, (poly[0][1] + poly[2][1]) // 2)
                    image = cv2.putText(image, f"{pi}-{label}", center, cv2.FONT_HERSHEY_SIMPLEX, 
                                        color=(0, 0, 255), fontScale=1)
            nimgs.append(image)

        self.imgs = nimgs

    def write(self, names: "list" = None):
        
        for idx, im in enumerate(self.imgs):
            if im is None:
                continue
            if names is None:
                loc = osp.join(self.img_root, f"gt_{idx + self.offset}.png")
            else:
                name = osp.basename(names[idx])
                name, ext = osp.splitext(name)
                loc = osp.join(self.img_root, f"{name}.png")
            cv2.imwrite(loc, im)
        return len(self.imgs)
            
    def plot(self, bboxes: "list[np.ndarray]", labels: "list[np.ndarray]",
            out_file=None, font_scale=0.6,
        ):

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        ## 
        # if "0086" not in img_meta["filename"]:
        #     continue
        if out_file is not None:
            img_loc = osp.join(self.img_root, out_file) # Test only
        else:
            out_file = ""
            img_loc = None
        img = cv2.imread(img)
        class_names = self.class_names
        class_colors = self.class_colors
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)[:-1].reshape(-1, 2)
            if label > len(class_names):
                label_text = "bg"
                continue
            else:
                label_text = class_names[label]
            offset = 0.0
            bbox_color = class_colors[label_text]

            text_color = bbox_color
            for i in range(4):
                cv2.line(img, bbox_int[i], bbox_int[(i+1) % 4], bbox_color, thickness=self.thickness)
            
            if len(bbox) > 8:
                label_text += '|{:.02f}'.format(bbox[-1]+ offset)

            tl = (bbox_int[0][0], bbox_int[0][1] - 2)
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, 3)
            text_w, text_h = text_size
            cv2.rectangle(img, (bbox_int[0][0], bbox_int[0][1] - text_h - 2), 
                        (bbox_int[0][0] + text_w, bbox_int[0][1] - 2), (255, 255, 255), -1)
            cv2.putText(img, label_text, tl,
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if self.storage_mode is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            text_color = (10,10,10)
            cv2.putText(img, out_file, (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        elif img_loc is not None:
            cv2.imwrite(img_loc, img,)

    def clear(self):
        pass
