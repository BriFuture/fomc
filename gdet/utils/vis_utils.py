import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image

def save_image(img_tensor: "torch.Tensor", boxes: "list[torch.Tensor]" = None, name="pic1.png", dst_dir='checkpoints/pic/'):
    img: "np.ndarray" = img_tensor.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    # print(img.shape, img.max(), img.min())
    im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        boxes = boxes.detach().cpu().numpy()
        num_box = len(boxes)
        for ni in range(num_box):
            box: "np.ndarray" = boxes[ni]
            # box = 
            box = box.astype(np.int32)
            if (box == 0).all():
                continue
            # print(box)
            cv2.rectangle(im, box[:2], box[2:], (0, 255, 255), 2)
            cv2.putText(im, f"{ni}", box[:2], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color= (0, 0, 255), thickness=1)
    dst = osp.join(dst_dir, name)
    status = cv2.imwrite(dst, im)
    print(dst, status)
def save_image_with_scores(img_tensor: "torch.Tensor", boxes: "torch.Tensor", 
    scores: "torch.Tensor", idxes: "torch.Tensor" = None,
    name="pic1.png", dst_dir='checkpoints/pic/'
):
    img: "np.ndarray" = img_tensor.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    # print(img.shape, img.max(), img.min())
    im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    num_box = len(boxes)
    for ni in range(num_box):
        box: "np.ndarray" = boxes[ni]
        score = scores[ni]
        if idxes is None:
            idx = 0
        else:
            idx = idxes[ni]
        # box = 
        box = box.astype(np.int32)
        if (box == 0).all():
            continue
        # print(box)
        cv2.rectangle(im, box[:2], box[2:], (0, 255, 255), 2)
        cv2.putText(im, f"{idx}-{score:.2f}", box[:2], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color= (0, 0, 255), thickness=1)
    dst = osp.join(dst_dir, name)
    status = cv2.imwrite(dst, im)
    print(dst, status)
def save_image_np(img: "np.ndarray", boxes: "list[torch.Tensor]" = None, 
                  name="pic2.png", dst_dir='checkpoints/pic/'):
    
    # img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    print(img.shape, img.max(), img.min())
    # im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        if type(boxes) is torch.Tensor:
            boxes = boxes.detach().cpu().numpy()
        num_box = len(boxes)
        for ni in range(num_box):
            box: "np.ndarray" = boxes[ni]
            # box = 
            box = box.astype(np.int32)
            if (box == 0).all():
                continue
            
            cv2.rectangle(img, box[:2], box[2:], (0, 255, 255), 2)
            cv2.putText(img, f"{ni}", box[:2], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color= (0, 0, 255), thickness=1)
    # im = Image.fromarray(img)
    dst = osp.join(dst_dir, name)
    # status = im.save(dst)
    status = cv2.imwrite(dst, img)
    print(dst, status)
    