import os, os.path as osp
from argparse import ArgumentParser
import pickle
import tqdm as tqdm
import numpy as np
import shutil as sh
import xml.etree.ElementTree as ET
import cv2
import torch
from dataclasses import dataclass
from bfcommon.config import Config
from bfcommon.utils import set_random_seed
from gdet.engine.exp_base import load_modules
from gdet.core.bbox import poly_iou_rotated_np
from gdet.registries import DATASETS
from gdet.core.bbox import  obb2poly_np, poly2obb_np

from fs.uitls import prettify_xml, hash_names

def construct_dataset(cfg: "DatasetConfigType"):
    data_cfg = cfg.clone()
    data_cls_type = data_cfg.pop('type')
    data_cls = DATASETS.get(data_cls_type)
    assert data_cls is not None, f"data_cls_type: {data_cls_type}"

    dataset = data_cls(data_cfg, transforms=None)
    dataset.init()
    return dataset

def construct_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_np", type=int, default=None)
    parser.add_argument("--prev_seed", type=int, default=0)
    parser.add_argument("--prev_shot", type=int, default=0)
    parser.add_argument("--skip_mask", action="store_true")
    parser.add_argument("--skip_select", action="store_true")
    parser.add_argument("--mask_unsel_shots", action="store_true")
    parser.add_argument("--remove_exist_seed", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    return parser

@dataclass
class DatasetInstance():
    id: int
    name: str
    bbox: list
    img_idx: int
    polygons: list
    filename: str = ""
    def __eq__(self, other):
        if isinstance(other, DatasetInstance):
            return self.id == other.id
        return False
SIGMAX = 15
class ShotSelector():
    IMG_SUFFIX = "jpg"
    def __init__(self, dataset, shot, seed):
        self.dataset = dataset
        self.select_mode = 'random'
        self.shot = shot
        self.seed = seed
        self.prev_seed = 0
        self.prev_shot = 0
        self.skip_mask = False
        self.skip_select = False
        self.mask_unsel_shots = False
        self.remove_exist_seed = False
        self.last_obj_hash = ""
        self.gaussian_kernel = (51, 51)

    def merge_objects(self, select_objects: "list[DatasetInstance]"):
        """将 objects 按照 img 进行合并
        """
        select_objects_by_img = {}
        all_img_ids = []
        filenames = []
        for so in select_objects:
            fn_key = (so.filename, so.img_idx)
            if fn_key not in filenames:
                filenames.append(fn_key)
            if so.img_idx not in select_objects_by_img:
                so_list = []
                select_objects_by_img[so.img_idx] = so_list
            else:
                so_list = select_objects_by_img[so.img_idx]
            so_list.append(so)
            all_img_ids.append(f"{so.id}")
        hash = hash_names(all_img_ids)
        self.last_obj_hash = hash
        print(f"Select object ids hash: {hash}")
        filenames = list(sorted(filenames))
        print(f"Select merged object images ({len(filenames)}): {filenames}")
        return select_objects_by_img

    def select(self):
        select_objects = []
        select_objects_by_cat = {}
        if self.select_mode == "random":
            objects_by_cat = self.select_all_objects(self.dataset)
            ## bsf.c 用之前用过的 shot
            curr_shot = self.shot - self.prev_shot

            for name, objects in objects_by_cat.items():
                if len(objects) <= curr_shot or self.skip_select:
                    sel_objs = objects
                else:
                    sel_objs = np.random.choice(objects, curr_shot, replace=False)
                select_objects_by_cat[name] = sel_objs
                select_objects.extend(sel_objs)
        
        if self.prev_shot != 0:
            prev_objects_file = self.get_prev_object_file_dir()
            prev_objects, prev_objs_by_cat = self.read_objects(prev_objects_file)
            print(f"Load previous shots: {len(prev_objects)}")
            for k, v in prev_objs_by_cat.items():
                curr_list = select_objects_by_cat[k].tolist()
                if type(v) == np.ndarray:
                    v = v.tolist()
                select_objects_by_cat[k] = curr_list + v
            select_objects.extend(prev_objects)
            pass

        select_obj_ids_by_cat = {}
        for c, objs in select_objects_by_cat.items():
            select_obj_ids_by_cat[c] = [obj.id for obj in objs]
        print(f"Select objects {len(select_obj_ids_by_cat)}: {select_obj_ids_by_cat}")

        select_objects_by_img = self.merge_objects(select_objects)
        if not self.skip_select:
            self.remove_seed_dir()
        object_file_loc = self.get_object_file_dir()
        self.write_objects(object_file_loc, select_objects, select_objects_by_cat)
        self.write_annotations(select_objects_by_img)

    def remove_seed_dir(self):
        pass

    def write_objects(self, file_dir, select_objects: "list[DatasetInstance]", select_objects_by_cat):
        file = osp.join(file_dir, "objects.pkl")
        print(f"Object instance ({len(select_objects)}) info dumps into: `{file}`")
        with open(file, "wb") as f:
            dat = {
                "select_objects": select_objects,
                "select_objects_by_cat": select_objects_by_cat,
            }
            pickle.dump(dat, f)
        with open(osp.join(file_dir, "object_ids.txt"), "w") as f:
            for sel_obj in select_objects:
                f.write(f"{sel_obj.id}\n")

    def read_objects(self, file_dir, ):
        file = osp.join(file_dir, "objects.pkl")
        with open(file, "rb") as f:
            dat: "dict" = pickle.load(f)
            select_objects: "list[DatasetInstance]" = dat['select_objects']
            select_objects_by_cat: "dict" = dat['select_objects_by_cat']
        return select_objects, select_objects_by_cat

    def write_annotations(self, select_objects_by_img: "dict"):
        pass

    def select_all_objects(self, train_dataset):
        
        label2name = {k: v for k, v in zip(train_dataset.m_catId2label.values(), train_dataset.m_catId2name.values(), )}
        objects_by_cat = {}
        for name in train_dataset.m_catId2name.values():
            objects_by_cat[name] = []
        for i in range(len(train_dataset)):
            di = train_dataset.m_data_infos[i]
            img_id = di['id']
            ann_info = train_dataset.load_ann_info(img_id, )
            bboxes = ann_info['polygons']
            labels = ann_info['labels']
            names = [label2name[lab] for lab in labels]
            ids = ann_info['ids']
            for name, oid, bbox in zip(names, ids, bboxes):
                di = DatasetInstance(id=oid, name=name, bbox=bbox, img_idx=i, polygons=bbox)
                objects_by_cat[name].append(di)
        return objects_by_cat

    def get_curr_seed_dir(self):
        return self.get_seed_shot_dir(self.seed, self.shot)

    def get_seed_shot_dir(self, seed, shot):
        seed_dir = f"seed{seed}_shot{shot}"
        if self.mask_unsel_shots:
            seed_dir = f"mask_seed{seed}_shot{shot}"
        return seed_dir
    
    def get_object_file_dir(self) :
        raise ValueError("Unsupported")
        return ""
    
    def get_prev_object_file_dir(self):
        curr_shot = self.shot
        curr_seed = self.seed

        self.shot = self.prev_shot
        self.seed = self.prev_seed
        file_loc = self.get_object_file_dir()
        self.shot = curr_shot
        self.seed = curr_seed
        return file_loc
    
class DIOR_ShotSelector(ShotSelector):
    def remove_seed_dir(self):
        train_dataset = self.dataset
        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)
        print(dst_dir)
        if osp.exists(dst_dir):
            if self.remove_exist_seed:
                sh.rmtree(dst_dir)
                # breakpoint()
            else:
                raise ValueError(f"Dst dir ({dst_dir}) has existed, please remove it or change another seed")
            
    def get_object_file_dir(self,):
        train_dataset = self.dataset
        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)

        imgset_dir  = osp.join(dst_dir, "ImageSets/Main")
        os.makedirs(imgset_dir, exist_ok=True)
        return imgset_dir
    
    def write_annotations(self, select_objects_by_img: "dict"):
        ### 将 select objects 写入到 指定目录
        train_dataset = self.dataset

        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)

        os.makedirs(dst_dir, exist_ok=True)
        annos_dir = osp.join(dst_dir, "Annotations")
        imgs_dir  = osp.join(dst_dir, "JPEGImages")
        imgset_dir  = osp.join(dst_dir, "ImageSets/Main")
        os.makedirs(imgset_dir, exist_ok=True)
        os.makedirs(annos_dir, exist_ok=True)
        os.makedirs(imgs_dir,  exist_ok=True)
        
        imgset_path = osp.join(imgset_dir, "train.txt")
        with open(imgset_path, "w") as f:
            f.write(f"# objects hash: {self.last_obj_hash}\n")
            for img_idx in sorted(select_objects_by_img.keys()):
                data_info = train_dataset.m_data_infos[img_idx]
                img_id = data_info['id']
                f.write(f"{img_id}\n")

        ### 保存图片和anno
        masked_count = {}
        pbar = tqdm.tqdm(select_objects_by_img.items())
        for img_idx, so_list in pbar:
            data_info = train_dataset.m_data_infos[img_idx]
            id_list = [so.id for so in so_list]
            img_id = data_info['id']
            xml_path = osp.join(train_dataset.m_img_dir, train_dataset.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            unselected_bbox = []
            selected_bbox = []
            for idx, obj in enumerate(root.findall(".//object"), start=1):
                id_element = obj.find("oid")
                id = int(id_element.text)
                bnd_box = obj.find('robndbox')
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
                poly = np.asarray(poly, dtype=np.int32)
                if id not in id_list:
                    ### 该 object 没有被选中为 shot
                    ignore_el = ET.Element("ignore")
                    ignore_el.text = "1"
                    obj.append(ignore_el)
                    unselected_bbox.append(poly)
                else:
                    selected_bbox.append(poly)

            formatted_xml = prettify_xml(root)
            dst_xml_path = osp.join(annos_dir, f'{img_id}.xml')
            # 保存格式化的 XML
            with open(dst_xml_path, "w", encoding="utf-8") as f:
                f.write(formatted_xml)
            # 保存图片
            img_path = osp.join(train_dataset.m_img_dir, train_dataset.img_subdir, f'{img_id}.{self.IMG_SUFFIX}')
            dst_img_path = osp.join(imgs_dir,  f'{img_id}.{self.IMG_SUFFIX}')
            img = cv2.imread(img_path)
            if not self.mask_unsel_shots:
                cv2.imwrite(dst_img_path, img)
                continue

            if len(selected_bbox) > 0 and len(unselected_bbox) > 0:
                ### filter unselected bbox if it has overlap with selected bbox
                selected_bbox = np.asarray(selected_bbox)
                unselected_bbox = np.asarray(unselected_bbox)
                ious = poly_iou_rotated_np(
                    selected_bbox,
                    unselected_bbox)
                
                max_iou = ious.max(axis=0)
                mask = max_iou < 0.01
                # if not mask.all():
                #     print(dst_img_path)
                unselected_bbox = unselected_bbox[mask]
            if len(unselected_bbox) == 0:
                cv2.imwrite(dst_img_path, img)
                continue
            masked_count[img_id] = len(unselected_bbox)
            ### mask unselected shot
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for ui, unsel_box in enumerate(unselected_bbox):
                mask.fill(0)
                unsel_box = unsel_box.reshape(-1, 2)
                # cv2.fillPoly(mask, [unsel_box], 255)
                # blurred = cv2.GaussianBlur(img, self.gaussian_kernel, SIGMAX)
                # img[mask == 255] = blurred[mask == 255]
                x_min, y_min = np.min(unsel_box, axis=0)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max, y_max = np.max(unsel_box, axis=0)
                sub_img = img[y_min:y_max, x_min:x_max].copy()
                sub_mask = mask[y_min:y_max, x_min:x_max]
                local_poly = unsel_box - np.array([x_min, y_min])  # 变换到局部坐标
                cv2.fillPoly(sub_mask, [local_poly], 255)

                # **对子图进行高斯模糊**
                blurred_sub_img = cv2.GaussianBlur(sub_img, self.gaussian_kernel, SIGMAX)

                # **仅将掩码为 255 的部分替换回原图**
                sub_img[sub_mask == 255] = blurred_sub_img[sub_mask == 255]
                img[y_min:y_max, x_min:x_max] = sub_img  # 写回原图
                pbar.set_description(f"masking boxes {ui}/{len(unselected_bbox)}")

            cv2.imwrite(dst_img_path, img)
        if len(masked_count):
            print(masked_count)

class HRSC_ShotSelector(ShotSelector):
    IMG_SUFFIX = "bmp"
    def remove_seed_dir(self):
        train_dataset = self.dataset
        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)
        print(dst_dir)
        if osp.exists(dst_dir):
            if self.remove_exist_seed:
                sh.rmtree(dst_dir)
                # breakpoint()
            else:
                raise ValueError(f"Dst dir ({dst_dir}) has existed, please remove it or change another seed")    
    def get_object_file_dir(self):
        train_dataset = self.dataset
        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)

        imgset_dir  = osp.join(dst_dir, "ImageSets")
        os.makedirs(imgset_dir, exist_ok=True)
        return imgset_dir

    def write_annotations(self, select_objects_by_img: "dict"):
        ### 将 select objects 写入到 指定目录
        train_dataset = self.dataset

        root_dir = osp.join(train_dataset.m_img_dir, "split")
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)
        
        os.makedirs(dst_dir, exist_ok=True)
        annos_dir = osp.join(dst_dir, "Annotations")
        imgs_dir  = osp.join(dst_dir, "AllImages")
        imgset_dir  = osp.join(dst_dir, "ImageSets")
        os.makedirs(imgset_dir, exist_ok=True)
        os.makedirs(annos_dir, exist_ok=True)
        os.makedirs(imgs_dir,  exist_ok=True)
        
        imgset_path = osp.join(imgset_dir, "train.txt")
        with open(imgset_path, "w") as f:
            f.write(f"# objects hash: {self.last_obj_hash}\n")
            for img_idx in sorted(select_objects_by_img.keys()):
                data_info = train_dataset.m_data_infos[img_idx]
                img_id = data_info['id']
                f.write(f"{img_id}\n")

        ### 保存图片和anno
        masked_count = {}
        pbar = tqdm.tqdm(select_objects_by_img.items())
        for img_idx, so_list in pbar:
            data_info = train_dataset.m_data_infos[img_idx]
            id_list = [so.id for so in so_list]
            img_id = data_info['id']
            xml_path = osp.join(train_dataset.m_img_dir, train_dataset.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            unselected_bbox = []
            unselected_rbox = []
            selected_bbox = []
            hrsc_objects_tree = root.find("HRSC_Objects")
            for idx, obj in enumerate(hrsc_objects_tree.findall("HRSC_Object"), start=1):
                id_element = obj.find("Object_ID")
                id = int(id_element.text)
                rbox = [[
                    float(obj.find('mbox_cx').text),
                    float(obj.find('mbox_cy').text),
                    float(obj.find('mbox_w').text),
                    float(obj.find('mbox_h').text),
                    float(obj.find('mbox_ang').text),
                    0,
                ]]
                rbox = np.asarray(rbox, dtype=np.float32)
                poly = obb2poly_np(rbox, 'le90')[0, :-1].astype(np.float32)

                if id not in id_list:
                    ### 该 object 没有被选中为 shot
                    ignore_el = ET.Element("ignore")
                    ignore_el.text = "1"
                    obj.append(ignore_el)
                    unselected_bbox.append(poly)
                    unselected_rbox.append(rbox)
                else:
                    selected_bbox.append(poly)

            formatted_xml = prettify_xml(root)
            dst_xml_path = osp.join(annos_dir, f'{img_id}.xml')
            # 保存格式化的 XML
            with open(dst_xml_path, "w", encoding="utf-8") as f:
                f.write(formatted_xml)
            # 保存图片
            img_path = osp.join(train_dataset.m_img_dir, train_dataset.img_subdir, f'{img_id}.{self.IMG_SUFFIX}')
            dst_img_path = osp.join(imgs_dir,  f'{img_id}.{self.IMG_SUFFIX}')
            img = cv2.imread(img_path)
            if not self.mask_unsel_shots:
                cv2.imwrite(dst_img_path, img)
                continue

            if len(selected_bbox) > 0 and len(unselected_bbox) > 0:
                ### filter unselected bbox if it has overlap with selected bbox
                selected_bbox = np.asarray(selected_bbox)
                unselected_bbox = np.asarray(unselected_bbox)
                ious = poly_iou_rotated_np(
                    selected_bbox,
                    unselected_bbox)
                
                max_iou = ious.max(axis=0)
                mask = max_iou < 0.01
                # if not mask.all():
                #     print(dst_img_path)
                unselected_bbox = unselected_bbox[mask]
            if len(unselected_bbox) == 0:
                cv2.imwrite(dst_img_path, img)
                continue
            masked_count[img_id] = len(unselected_bbox)
            ### mask unselected shot
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for ui, unsel_box in enumerate(unselected_bbox):
                mask.fill(0)
                unsel_box = unsel_box.reshape(-1, 2).astype(np.int32)
                
                # cv2.fillPoly(mask, [unsel_box], 255)
                # blurred = cv2.GaussianBlur(img, self.gaussian_kernel, SIGMAX)
                # img[mask == 255] = blurred[mask == 255]
                
                x_min, y_min = np.min(unsel_box, axis=0)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max, y_max = np.max(unsel_box, axis=0)
                sub_img = img[y_min:y_max, x_min:x_max].copy()
                sub_mask = mask[y_min:y_max, x_min:x_max]
                local_poly = unsel_box - np.array([x_min, y_min])  # 变换到局部坐标
                cv2.fillPoly(sub_mask, [local_poly], 255)

                # **对子图进行高斯模糊**
                blurred_sub_img = cv2.GaussianBlur(sub_img, self.gaussian_kernel, SIGMAX)

                # **仅将掩码为 255 的部分替换回原图**
                sub_img[sub_mask == 255] = blurred_sub_img[sub_mask == 255]
                img[y_min:y_max, x_min:x_max] = sub_img  # 写回原图
                pbar.set_description(f"masking boxes {ui}/{len(unselected_bbox)}")
            # for unsel_box in unselected_bbox:
            #     unsel_box = unsel_box.reshape(-1, 2).astype(np.int32)
            #     cv2.polylines(img, [unsel_box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite(dst_img_path, img)
        if len(masked_count):
            print(masked_count)


class DOTA_ShotSelector(ShotSelector):
    IMG_SUFFIX = "png"
    IGNORE_DIFFICULTY = 110
    def select_all_objects(self, train_dataset):
        
        label2name = {k: v for k, v in zip(train_dataset.m_catId2label.values(), train_dataset.m_catId2name.values(), )}
        objects_by_cat = {}
        for name in train_dataset.m_catId2name.values():
            objects_by_cat[name] = []
        assert len(train_dataset) > 0
        for i in range(len(train_dataset)):
            di = train_dataset.m_data_infos[i]
            ann_info = di['ann']
            bboxes = ann_info['bboxes']
            polygons = ann_info['polygons']
            labels = ann_info['labels']
            names = [label2name[lab] for lab in labels]
            ids = ann_info['ids']
            for name, oid, bbox, poly in zip(names, ids, bboxes, polygons):
                ins = DatasetInstance(id=oid, name=name, bbox=bbox, img_idx=i, polygons=poly)
                ins.filename = di['filename']
                objects_by_cat[name].append(ins)
        return objects_by_cat

    def remove_seed_dir(self):
        train_dataset = self.dataset
        ds_root_dir = osp.dirname(train_dataset.m_img_dir)
        root_dir = osp.join(ds_root_dir, "split")
        
        seed_dir = self.get_curr_seed_dir()
        dst_dir = osp.join(root_dir, seed_dir)
        print(dst_dir)
        if osp.exists(dst_dir):
            if self.remove_exist_seed:
                sh.rmtree(dst_dir)
                # breakpoint()
            else:
                raise ValueError(f"Dst dir ({dst_dir}) has existed, please remove it or change another seed")
                    
    def get_object_file_dir(self):
        train_dataset = self.dataset
        if self.skip_select:
            ds_root_dir = osp.dirname(train_dataset.m_img_dir)
            root_dir = ds_root_dir
            dst_dir = root_dir
        else:
            ds_root_dir = osp.dirname(train_dataset.m_img_dir)
            root_dir = osp.join(ds_root_dir, "split")
            seed_dir = self.get_curr_seed_dir()
            dst_dir = osp.join(root_dir, seed_dir)
        
        imgset_dir  = osp.join(dst_dir, "ImageSets")
        
        os.makedirs(imgset_dir, exist_ok=True)
        return imgset_dir

    def write_annotations(self, select_objects_by_img: "dict"):
        ### 将 select objects 写入到 指定目录
        train_dataset = self.dataset
        ds_root_dir = osp.dirname(train_dataset.m_img_dir)
        if self.skip_select:
            root_dir = ds_root_dir
            dst_dir = root_dir
        else:
            root_dir = osp.join(ds_root_dir, "split")
            seed_dir = self.get_curr_seed_dir()
            dst_dir = osp.join(root_dir, seed_dir)

        os.makedirs(dst_dir, exist_ok=True)
        annos_dir = osp.join(dst_dir, "labelTxt")
        imgs_dir  = osp.join(dst_dir, "images")
        imgset_dir  = osp.join(dst_dir, "ImageSets")
        os.makedirs(imgset_dir, exist_ok=True)
        os.makedirs(annos_dir, exist_ok=True)
        os.makedirs(imgs_dir,  exist_ok=True)
        
        imgset_path = osp.join(imgset_dir, "train.txt")
        print(f"\n\nsave imageset into {imgset_path}, annos: {annos_dir} ")
        with open(imgset_path, "w") as f:
            f.write(f"# objects hash: {self.last_obj_hash}\n")
            for img_idx in sorted(select_objects_by_img.keys()):
                data_info = train_dataset.m_data_infos[img_idx]
                img_id = data_info['file_id']
                f.write(f"{img_id}\n")

        ### 保存图片和anno
        masked_count = {}
        pbar = tqdm.tqdm(select_objects_by_img.items())
        for img_idx, so_list in pbar:
            data_info = train_dataset.m_data_infos[img_idx]
            id_list = [so.id for so in so_list]
            img_id = data_info['file_id']
            txt_path = osp.join(train_dataset.ann_folder, f'{img_id}.txt')
            unselected_bbox = []
            unselected_rbox = []
            selected_bbox = []
            all_objects = []
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0: continue
                    items = line.split(" ")
                    if len(items) >= 10:
                        bbox = [float(i) for i in items[:8]]
                        id_item = items[10][3:]
                        diff_item = int(items[9])
                        current_object = {
                            "box": bbox, "label": items[8], "diff": diff_item,
                            "id": int(id_item)
                        }
                        if current_object['id'] not in id_list:
                            current_object['diff'] = self.IGNORE_DIFFICULTY
                            unselected_bbox.append(bbox)
                            # unselected_rbox.append(rbox)
                        else:
                            selected_bbox.append(bbox)
                        all_objects.append(current_object)
                    else:
                        # raise ValueError(f"Please id objects first: {line}")
                        pass
            if not self.skip_select:
                dst_txt_path = osp.join(annos_dir, f'{img_id}.txt')
                with open(dst_txt_path, "w", encoding="utf-8") as f:
                    for obj in all_objects:
                        outline = ' '.join(list(map(str, obj['box'])))
                        diffs = str(obj['diff']) 
                        outline = outline + ' ' + obj['label'] + ' ' + diffs
                        outline += ' ' + f"id:{obj['id']}"
                        f.write(outline + '\n')
            # 读取原图
            img_path = osp.join(train_dataset.m_img_dir, f'{img_id}.{self.IMG_SUFFIX}')
            dst_img_path = osp.join(imgs_dir,  f'{img_id}.{self.IMG_SUFFIX}')
            img = cv2.imread(img_path)
            if img is None:
                print(img_path)
            # 保存图片
            if not self.mask_unsel_shots:
                cv2.imwrite(dst_img_path, img)
                continue

            if len(selected_bbox) > 0 and len(unselected_bbox) > 0:
                ### filter unselected bbox if it has overlap with selected bbox
                selected_bbox = np.asarray(selected_bbox)
                unselected_bbox = np.asarray(unselected_bbox)
                ious = poly_iou_rotated_np(
                    selected_bbox,
                    unselected_bbox)
                
                max_iou = ious.max(axis=0)
                mask = max_iou < 0.01
                # if not mask.all():
                #     print(dst_img_path)
                unselected_bbox = unselected_bbox[mask]
            if len(unselected_bbox) == 0:
                cv2.imwrite(dst_img_path, img)
                continue
            masked_count[img_id] = len(unselected_bbox)
            ### mask unselected shot
            if not self.skip_mask:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for ui, unsel_box in enumerate(unselected_bbox):
                    mask.fill(0)
                    # unsel_box = unsel_box.reshape(-1, 2).astype(np.int32)
                    # cv2.fillPoly(mask, [unsel_box], 255)
                    # # 复制原图并进行高斯模糊
                    # blurred = cv2.GaussianBlur(img, self.gaussian_kernel, SIGMAX)
                    # img[mask == 255] = blurred[mask == 255]
                    unsel_box = unsel_box.reshape(-1, 2).astype(np.int32)
                    x_min, y_min = np.min(unsel_box, axis=0)
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max, y_max = np.max(unsel_box, axis=0)
                    sub_img = img[y_min:y_max, x_min:x_max].copy()
                    sub_mask = mask[y_min:y_max, x_min:x_max]
                    if sub_mask.shape[0] == 0 or sub_mask.shape[1] == 0:
                        # print(img_path, unsel_box)
                        continue
                    local_poly = unsel_box - np.array([x_min, y_min])  # 变换到局部坐标
                    cv2.fillPoly(sub_mask, [local_poly], 255)

                    # **对子图进行高斯模糊**
                    blurred_sub_img = cv2.GaussianBlur(sub_img, self.gaussian_kernel, SIGMAX)

                    # **仅将掩码为 255 的部分替换回原图**
                    sub_img[sub_mask == 255] = blurred_sub_img[sub_mask == 255]
                    img[y_min:y_max, x_min:x_max] = sub_img  # 写回原图
                    pbar.set_description(f"masking boxes {ui}/{len(unselected_bbox)}")

            # for unsel_box in unselected_bbox:
            #     unsel_box = unsel_box.reshape(-1, 2).astype(np.int32)
            #     cv2.polylines(img, [unsel_box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite(dst_img_path, img)
        if len(masked_count):
            print(masked_count)


def main(args=None):
    parser = construct_parser()
    args = parser.parse_args(args)
    config_path = args.config
    shot = args.shot
    seed = args.seed
    seed_np = args.seed_np
    print(args)
    if seed is None:
        seed = shot
    if seed_np is None:
        seed_np = seed
    set_random_seed(seed_np)
    config = Config.fromfile(config_path)
    load_modules(config.dataset)
    cfg_train = config.dataset["train"].clone()
    # cfg_prev = config.dataset["train"].clone()
    if args.dataset == "dior":
        train_dataset = construct_dataset(cfg_train)
        shotsel = DIOR_ShotSelector(train_dataset, shot, seed)
    elif args.dataset == "hrsc":
        train_dataset = construct_dataset(cfg_train)
        shotsel = HRSC_ShotSelector(train_dataset, shot, seed)
    elif args.dataset == "dota":
        if 'split' in cfg_train.ann_file:
            _data_root = f"datasets/DOTA_v1/train_ss/split/mask_seed{seed}_shot{shot}/"
            cfg_train['ann_file'] = _data_root+"ImageSets/Main/train.txt"
            cfg_train['img_dir'] = _data_root + "images"
            cfg_train['ann_folder'] = _data_root+"annfiles"
        train_dataset = construct_dataset(cfg_train)
        shotsel = DOTA_ShotSelector(train_dataset, shot, seed)
    else:
        raise ValueError(f"dataset not supported {args.dataset}")

    shotsel.prev_shot = args.prev_shot
    shotsel.prev_seed = args.prev_seed

    shotsel.mask_unsel_shots = args.mask_unsel_shots
    shotsel.skip_mask = args.skip_mask
    shotsel.skip_select = args.skip_select
    shotsel.remove_exist_seed   = args.remove_exist_seed

    shotsel.select()

import sys
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = ["--shot=5", "--seed=6", "--remove_exist_seed", "--mask_unsel_shots", "--config=configs/dior_r/orcnn/split1/ds_orcnn_shot.py"]
        args = ["--shot=10", "--seed=5", "--remove_exist_seed", 
                # "--mask_unsel_shots", 
                "--dataset=hrsc", "--config=configs/hrsc/orcnn/split1/ds_orcnn_shot.py"
            ]
        args = ["--shot=10", "--seed=6", "--remove_exist_seed", 
                # "--mask_unsel_shots", 
                "--dataset=dota", "--config=configs/dota/ds_dota_origin.py"
            ]
        args = ["--shot=3", "--seed=8", "--remove_exist_seed", "--mask_unsel_shots", "--dataset=dior", "--config=configs/dior_r/ds_dior.py"]
        args = ["--shot=20", "--seed=5", "--mask_unsel_shots", "--skip_select", "--dataset=dota", "--config=configs/dota/fomc/ds_dota_origin.py"]
    main(args)
