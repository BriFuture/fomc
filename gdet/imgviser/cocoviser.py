from .baseviser import Visualizer
from pycocotools.coco import COCO
from skimage import io
import numpy as np
import json, os
from matplotlib import pyplot as plt
import math
import cv2
from fs.core.bbox.transforms_rotated import rotated_box_to_poly_single

class CocoVisualizer(Visualizer):
    
    def __init__(self, basedir, json_path) -> None:
        super().__init__(basedir)
        self.coco = COCO(json_path)

    def vis(self, num_image):
        coco = self.coco
        catIds = coco.getCatIds()
        list_imgIds = coco.getImgIds(catIds=catIds) # 获取含有该给定类别的所有图片的id
        img = coco.loadImgs(list_imgIds[num_image-1])[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
        print("imageids", img)
        image = io.imread(self.imgdir + img['file_name'])  # 读取图像
        image_name =  img['file_name'] # 读取图像名字
        image_id = img['id'] # 读取图像id
        img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
        img_anns = coco.loadAnns(img_annIds)
        # print(img_annIds)
        for i in range(len(img_annIds)):
            ann = img_anns[i]
            cid = ann["category_id"]

            print("box: ", i, ann['bbox']) 
            box = ann['bbox']
            box[-1] = box[-1] / 180 * math.pi
            box = rotated_box_to_poly_single(ann['bbox'])
            poly = np.array(box, dtype=np.int32)
            poly = poly.reshape((-1, 2))
            print(poly)
            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1],(255,0,0), 2, lineType=cv2.LINE_AA)
            image = cv2.line(image, poly[3], poly[0],(255,0,0), 2, lineType=cv2.LINE_AA)
            print(" === \n", poly, cid)
            # break
        plt.imshow(image)
        plt.show()


