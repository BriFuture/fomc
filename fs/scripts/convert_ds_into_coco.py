from bfcommon.config import Config
from bfcommon.logger import setup_logger
import os, os.path as osp
import copy, sys
from gdet.data_factory import construct_dataset
from gdet.datasets.dataset.coco_dataset import CocoDataset
from gdet.datasets.dataset.voc_dataset import VocDataset
from gdet.datasets.dataset.dota import DOTADataset
from gdet.engine.exp_base import load_modules
from PIL import Image
import cv2
import tqdm
import json
from collections import Counter
import numpy as np
from argparse import ArgumentParser
CURR_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(CURR_PATH, "..", ".."))

def construct_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--data_type", type=str, default="val")
    return parser

setup_logger(name="gdet")

def convert_voc_into_coco(args=None,
):
    """bsf.c 通过读取 VOC 数据集，将其转为 COCO 数据集
    """

    # cfg = Config.fromfile(osp.join(ROOT_PATH, "configs/dior/ds_dior_r_voc.py"))
    cfg = Config.fromfile(osp.join(ROOT_PATH, args.config))
    data_type = args.data_type

    train_cfg = cfg.dataset[data_type]
    full_cfg = copy.deepcopy(train_cfg)

    full_cfg.dst_classes = cfg.dataset.train.dst_classes

    voc_dataset: "VocDataset" = construct_dataset(full_cfg)
    print(f"Full: {len(voc_dataset)}")

    ### convert into coco format
    coco_json = {
        "info": {
            "description": "DIOR Dataset",
            "version": "1.0",
        },
    }
    images = []
    for di in voc_dataset.m_data_infos:
        filename = osp.basename(di['filename'])
        image_item = {
            "file_name": filename,
            "height": di['height'],
            "width": di['width'],
            "id": int(di['id']),
        }
        images.append(image_item)
    coco_json['images'] = images

    # anno_id = 0
    anno_items = []
    for i in range(len(voc_dataset)):
        ann = voc_dataset.get_ann_info(i)
        img_id = voc_dataset.m_data_infos[i]['id']
        img_id = int(img_id)
        bbox = ann['bboxes'][0]
        bbox = bbox.astype(int).tolist()
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        segment = [
            bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3],
        ]
        obj_id = int(ann['ids'][0])
        ann_item = {
            "area": area,
            "category_id": int(ann['labels'][0]),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": bbox,
            "segmentation": [segment],
            "id": obj_id,
        }
        anno_items.append(ann_item)
    coco_json['annotations'] = anno_items

    cat_items = []
    for id, cat in voc_dataset.m_catId2name.items():
        cat_item = {
            "supercategory": cat,
            "id": id + 1,
            "name": cat
        }
        cat_items.append(cat_item)
    coco_json['categories'] = cat_items

    coco_ann_dir = osp.join(ROOT_PATH, voc_dataset.m_img_dir, "coco_annos")
    os.makedirs(coco_ann_dir, exist_ok=True)
    dst_file = osp.join(coco_ann_dir, f"{data_type}.json")
    with open(dst_file, "w") as f:
        json.dump(coco_json, f, indent=2)


def convert_hrsc_into_coco(args=None,
):
    """bsf.c 通过读取 VOC 数据集，将其转为 COCO 数据集
    """

    # cfg = Config.fromfile(osp.join(ROOT_PATH, "configs/dior/ds_dior_r_voc.py"))
    cfg = Config.fromfile(osp.join(ROOT_PATH, args.config))
    load_modules(cfg.dataset, hint=True)
    data_type = args.data_type

    train_cfg = cfg.dataset[data_type]
    full_cfg = copy.deepcopy(train_cfg)

    full_cfg.dst_classes = cfg.dataset.train.all_classes

    voc_dataset: "VocDataset" = construct_dataset(full_cfg)
    print(f"Dataset Full Count: {len(voc_dataset)}")

    ### convert into coco format
    coco_json = {
        "info": {
            "description": "DIOR Dataset",
            "version": "1.0",
        },
    }
    images = []
    for di in voc_dataset.m_data_infos:
        filename = osp.basename(di['filename'])
        image_item = {
            "file_name": filename,
            "height": di['height'],
            "width": di['width'],
            "id": int(di['id']),
        }
        images.append(image_item)
    coco_json['images'] = images

    # anno_id = 0
    anno_items = []
    category_count = Counter()
    for i in range(len(voc_dataset)):
        ann = voc_dataset.get_ann_info(i)
        img_id = voc_dataset.m_data_infos[i]['id']
        img_id = int(img_id)
        bboxes = ann['bboxes'][0]
        # bboxes = bboxes.astype(int).tolist()

        poly: "np.ndarray" = ann['polygons'][0]
        bboxes = poly.astype(int).tolist()
        bbox = np.zeros((4,), dtype=np.float32)
        bbox[0] = min(poly[0::2])
        bbox[1] = min(poly[1::2])
        bbox[2] = max(poly[0::2])
        bbox[3] = max(poly[1::2])

        bbox = bbox.astype(int).tolist()
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        bbox_xywh = [bbox[0], bbox[1], w, h]
        area = (w + 1) * (h + 1)
        segment = poly.astype(int).tolist()
        obj_id = int(ann['ids'][0])
        cat_id = int(ann['labels'][0]) + 1
        category_count[cat_id] += 1
        ann_item = {
            "area": area,
            "category_id": cat_id,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": bboxes,
            "segmentation": [segment],
            "id": obj_id,
        }
        anno_items.append(ann_item)
    
    category_counts = {k: category_count[k] for k in sorted(category_count.keys())}
    print("Category counts: ", category_counts)
    coco_json['annotations'] = anno_items

    cat_items = []
    cat_id = 1
    for id, cat in voc_dataset.m_catId2name.items():
        id = int(id)
        cat_item = {
            "supercategory": cat,
            "id": cat_id ,
            "name": cat
        }
        cat_id += 1
        cat_items.append(cat_item)
    coco_json['categories'] = cat_items

    coco_ann_dir = osp.join(ROOT_PATH, voc_dataset.m_img_dir, "coco_annos")
    os.makedirs(coco_ann_dir, exist_ok=True)
    dst_file = osp.join(coco_ann_dir, f"{data_type}.json")
    print(f"save coco annos into {dst_file}")
    with open(dst_file, "w") as f:
        json.dump(coco_json, f, indent=2)


def convert_dota_into_coco(
    args
):
    """bsf.c 通过读取 VOC 数据集，将其转为 COCO 数据集
    """
    setup_logger(name="gdet")
    cfg = Config.fromfile(osp.join(ROOT_PATH, args.config))

    load_modules(cfg.dataset, hint=True)

    data_type = args.data_type
    train_cfg = cfg.dataset[data_type]
    full_cfg = copy.deepcopy(train_cfg)

    full_cfg.dst_classes = cfg.dataset.train.dst_classes

    dota_dataset: "DOTADataset" = construct_dataset(full_cfg)
    print(f"Dota Full Images: {len(dota_dataset)}")

    ### convert into coco format
    coco_json = {
        "info": {
            "description": "DOTA Dataset",
            "version": "1.0",
        },
    }
    images = []
    g_img_id = 1
    anno_items = []
    g_obj_id = 1
    # for di in dota_dataset.m_data_infos:
    label_counter = Counter()
    for i in range(len(dota_dataset)):
        di = dota_dataset.m_data_infos[i]
        filename = osp.basename(di['filename'])
        file_loc = osp.join(dota_dataset.m_img_dir, filename)
        img = Image.open(file_loc)
        num_id = True
        if 'id' in di:
            try:
                img_id = int(di['id'])
            except :
                num_id = False
        else:
            num_id = False
        if not num_id:
            raise ValueError("Please check image id as int")
            img_id = g_img_id
            g_img_id += 1

        image_item = {
            "file_name": filename,
            "height": img.height,
            "width": img.width,
            "id": img_id,
        }
        images.append(image_item)

        ann = di['ann']
        # img_id = dota_dataset.m_data_infos[i]['id']
        polygons = ann['polygons']
        labels = ann['labels']
        for poly, label in zip(polygons, labels):
            segment = poly[:].tolist()
            bbox = np.zeros((4,), dtype=np.float32)
            bbox[0] = min(poly[0::2])
            bbox[1] = min(poly[1::2])
            bbox[2] = max(poly[0::2])
            bbox[3] = max(poly[1::2])

            bbox = bbox.astype(int).tolist()
            area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            obj_id = int(ann['ids'][0])
            if obj_id == 0:
                obj_id = g_obj_id
                g_obj_id += 1
            ### coco 映射时 bg 设为 0
            label += 1
            label_counter[label] += 1
            ann_item = {
                "area": area,
                "category_id": int(label),
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": bbox,
                "segmentation": [segment],
                "id": obj_id,
            }
            anno_items.append(ann_item)
    coco_json['images'] = images
    coco_json['annotations'] = anno_items

    cat_items = []
    for id, cat in dota_dataset.m_catId2name.items():
        cat_item = {
            "supercategory": cat,
            "id": id + 1,
            "name": cat
        }
        cat_items.append(cat_item)
    coco_json['categories'] = cat_items
    coco_ann_dir = osp.abspath(osp.join(ROOT_PATH, dota_dataset.m_img_dir, "..", "coco_annos"))
    os.makedirs(coco_ann_dir, exist_ok=True)
    dst_file = osp.join(coco_ann_dir, f"{data_type}.json")
    print(f"save coco annos into {dst_file}")
    print(f"Len images: {len(images)} annos: {len(anno_items)}  {g_obj_id}")
    label_count = {k: label_counter[k] for k in sorted(label_counter.keys())}
    print(f"Labels: {label_count}")
    print(f"Cat Items: \n{json.dumps(cat_items, indent=2)}")
    with open(dst_file, "w") as f:
        json.dump(coco_json, f, indent=2)

def convert_dataset(args=None):
    parser = construct_parser()
    args = parser.parse_args(args)
    if args.dataset == "voc":
        convert_voc_into_coco(args)
    elif args.dataset == "dota":
        convert_dota_into_coco(args)
    elif args.dataset == "hrsc":
        convert_hrsc_into_coco(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        # args = ["--dataset=voc", "--config=configs/dior/ds_dior_r_voc.py", "--data_type=val"]
        # args = ["--dataset=hrsc", "--config=configs/hrsc/orcnn/split1/ds_orcnn.py", "--data_type=val"]
        args = ["--dataset=dota", "--config=configs/dota/s2anet/split1/ds_s2anet.py", "--data_type=val_base"]
        args = ["--dataset=dota", "--config=configs/dota/s2anet/split1/ds_s2anet_shot.py", "--data_type=val"]
    convert_dataset(args=args)
    
