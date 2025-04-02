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
class BBoxViser(BaseViser):

    def plot(self, img, bboxes, labels, 
            out_file=None, vis=None, center=False):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 8) or
                (n, 9).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
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
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)[:-1].reshape(-1, 2)
            label_text = class_names[label]
            # if class_color is not None:
            bbox_color = class_colors[label_text]
            text_color = bbox_color
            # for i in range(4):
            #     cv2.line(img, bbox_int[i], bbox_int[(i+1) % 4], bbox_color, thickness=thickness)
            cv2.rectangle( img, bbox_int[0], bbox_int[1], bbox_color, thickness=self.thickness)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            tl = (bbox_int[0][0], bbox_int[0][1] - 2) if not center else ((bbox_int[0][0] + bbox_int[1][0]) // 2, (bbox_int[0][1] + bbox_int[1][1]) // 2)
            cv2.putText(img, label_text, tl,
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if vis:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.putText(img, out_file, (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        elif out_file is not None:
            cv2.imwrite(out_file, img,)
def box_to_poly_np(boxes: "np.ndarry"):
    polygons = []
    for box in boxes:
        poly = [
            [box[0], box[1]],
            [box[0], box[3]],
            [box[2], box[3]],
            [box[2], box[1]],
        ]
        poly = np.asarray(poly)
        polygons.append(poly)
    return np.asarray(polygons)

@VISER.register_module()
class GtBboxViser(BaseViser):
    def init(self, img_root = None):
        if img_root is None:
            img_root = "checkpoints/gt_bbox"
        self.img_root = img_root
        os.makedirs(self.img_root, exist_ok=True)
        ### for one batch, there may be several images
        self.imgs = []
        self.offset = 0
        self.overwrite = True
        self.fake_box = False
        
    def set_images(self, ximgs: "torch.Tensor", scales=None):
        self.imgs = []
        for imidx, xim in enumerate(ximgs):
            image, m1, m2 = convert_tensor_as_npint8(xim)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if scales is not None:
                scale = scales[imidx]
                hr, wr = scale.detach().cpu().numpy()
                original_height, original_width = image.shape[:2]
                new_width = int(original_width / wr)
                new_height = int(original_height / hr)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            self.imgs.append(image)

    def cplot(self, gt_bboxes: "list[torch.Tensor]",  color=(0, 0, 225)):
        nimgs = []
        for imidx, image in enumerate(self.imgs):
            if self.fake_box:
                nimgs.append(image)
                continue
            bbox = gt_bboxes[imidx]
            ### bsf.c bbox 可能为空，为空时画出原图
            if bbox is None:
                nimgs.append(image)
                continue
            bbox = bbox.detach().cpu().numpy()
            polygons = box_to_poly_np(bbox)
            for pi, poly in enumerate(polygons):
                poly = np.array(poly, dtype=np.int32)
                poly = poly.reshape((-1, 2))
                for i in range(3):
                    image = cv2.line(image, poly[i], poly[i+1], color, 1, lineType=cv2.LINE_AA)
                image = cv2.line(image, poly[3], poly[0], color, 3, lineType=cv2.LINE_AA)
                center = ((poly[0][0] + poly[2][0]) // 2, (poly[0][1] + poly[2][1]) // 2)
                image = cv2.putText(image, f"{imidx}-{pi}", center, cv2.FONT_HERSHEY_SIMPLEX, 
                                    color=(0, 0, 0), fontScale=1)
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
    
    def write_prediction(self, preds: "list[dict]", names: "list" = None):
        import json
        for idx in range(len(self.imgs)):
            if names is None:
                loc = osp.join(self.img_root, f"pd_{idx + self.offset}.txt")
            else:
                name = osp.basename(names[idx])
                name, ext = osp.splitext(name)
                loc = osp.join(self.img_root, f"{name}.txt")
            pd = preds[idx]
            with open(loc, "w") as f:
                gb_conf_mat = {}
                for gb, value in pd.items():
                    gt_cls = value[0]
                    f.write(f"gt [cls: {gt_cls}] {int(gb[0])},{int(gb[1])},{int(gb[2])},{int(gb[3])} \n")
                    pd_box = value[1]
                    if pd_box is None:
                        f.write("\n\n")
                        continue
                    pd_cls = value[2]
                    if pd_cls.ndim == 0:
                        pc = int(pd_cls)
                        pb = pd_box
                        f.write(f"  {0} [{pc}] {int(pb[0])},{int(pb[1])},{int(pb[2])},{int(pb[3])}  score: {value[3]:.3f} \n")
                    else:
                        for pi, pc in enumerate(pd_cls):
                            pb = pd_box[pi]
                            f.write(f"  {pi} [{pc}] {int(pb[0])},{int(pb[1])},{int(pb[2])},{int(pb[3])}  score: {value[3]:.3f} \n")
                    f.write("\n\n")
                    if gt_cls not in gb_conf_mat:
                        gbm = []
                        gb_conf_mat[gt_cls] = gbm
                    else:
                        gbm = gb_conf_mat[gt_cls]
                    if pd_cls.ndim == 0:
                        gbm.append(int(pd_cls))
                    else:
                        gbm.extend(list(int(i) for i in pd_cls))

                
                f.write("----\n")
                f.write(json.dumps(gb_conf_mat))
                f.write("\n")

        pass
        

    def plot(self, ximgs: "torch.Tensor", gt_bboxes: "list[torch.Tensor]", 
                            gt_labels: "list[torch.Tensor]" = None, offset: "int" = 0, color=(0, 0, 225)):
        """save_bimg_with_gtbbox
        xim: T[B, 3, h, w]
        gt_bboxes  list(xywha)
        offset: img_offset
        """
        for imidx, xim in enumerate(ximgs):
            bbox = gt_bboxes[imidx]
            bbox = bbox.detach().cpu().numpy()
            polygons = box_to_poly_np(bbox)
            
            image, m1, m2 = convert_tensor_as_npint8(xim)
            H, W = xim.shape[-2:]
            loc = self.get_im_loc((H, W), imidx + offset)
            if gt_labels is not None:
                gt_label = gt_labels[imidx]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for pi, poly in enumerate(polygons):
                poly = np.array(poly, dtype=np.int32)
                poly = poly.reshape((-1, 2))
                for i in range(3):
                    image = cv2.line(image, poly[i], poly[i+1], color, 1, lineType=cv2.LINE_AA)
                image = cv2.line(image, poly[3], poly[0], color, 3, lineType=cv2.LINE_AA)
                center = ((poly[0][0] + poly[2][0]) // 2, (poly[0][1] + poly[2][1]) // 2)
                if gt_labels is not None:
                    lb = f" - {gt_label[pi].item()}"
                else:
                    lb = ""
                image = cv2.putText(image, f"{pi}{lb}", center, cv2.FONT_HERSHEY_SIMPLEX, 
                                    color=(0, 0, 0), fontScale=0.5)
            if osp.exists(loc):
                if self.overwrite:
                    cv2.imwrite(loc, image)
            else:
                cv2.imwrite(loc, image)
        self.offset += len(ximgs)