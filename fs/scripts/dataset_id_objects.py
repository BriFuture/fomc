from bfcommon.config import Config
from gdet.data_factory import construct_dataset
import os, os.path as osp
import xml.etree.ElementTree as ET
from gdet.engine.exp_base import load_modules
import tqdm

from fs.uitls import prettify_xml

from argparse import ArgumentParser
def construct_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--config", type=str, default="configs/dior_r/ds_dior_origin.py")
    # parser.add_argument("--config", type=str, default="configs/hrsc/ds_hrsc.py")
    # parser.add_argument("--config", type=str, default="configs/dota/ds_dota_origin.py")
    parser.add_argument("--dataset", type=str, required=True)
    return parser


def dior_id_objects(dataset, g_id = 1):
    dst_folder_name = "Annotations"
    dst_folder = osp.join(dataset.m_img_dir, dst_folder_name)
    os.makedirs(dst_folder, exist_ok=True)
    
    for idx in tqdm.tqdm(range(len(dataset.m_data_infos))):
        img_id = dataset.m_data_infos[idx]['id']
        xml_path = osp.join(dataset.m_img_dir, dataset.ann_subdir, f'{img_id}.xml')
        dst_xml_path = osp.join(dst_folder, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
    
        # 找到所有 object 下的 bndbox，并添加 id
        for idx, obj in enumerate(root.findall(".//object"), start=1):
            id_element = ET.Element("oid")
            id_element.text = str(g_id)
            obj.append(id_element)
            # obj.set("id", str(g_id))  # 添加 id 属性
            g_id += 1        

        formatted_xml = prettify_xml(root)

        # 保存格式化的 XML
        with open(dst_xml_path, "w", encoding="utf-8") as f:
            f.write(formatted_xml)
    print(g_id)
    return g_id

def main_dior(args):
    config_path = args.config

    config = Config.fromfile(config_path)
    load_modules(config.dataset)
    train_dataset = construct_dataset(config.dataset['train'])
    g_id = 1
    new_id = dior_id_objects(train_dataset, g_id)
    g_id = new_id

    val_dataset = construct_dataset(config.dataset['val'])
    new_id = dior_id_objects(val_dataset, g_id)

def hrsc_id_objects(dataset, g_id = 1):
    dst_folder_name = "Annotations"
    dst_folder = osp.join(dataset.m_img_dir, dst_folder_name)
    os.makedirs(dst_folder, exist_ok=True)
    
    for idx in tqdm.tqdm(range(len(dataset.m_data_infos))):
        img_id = dataset.m_data_infos[idx]['id']
        xml_path = osp.join(dataset.m_img_dir, dataset.ann_subdir, f'{img_id}.xml')
        dst_xml_path = osp.join(dst_folder, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        hrsc_objects_tree = root.find("HRSC_Objects")
        # 找到所有 object 下的 bndbox，并添加 id
        for idx, obj in enumerate(hrsc_objects_tree.findall(".//HRSC_Object"), start=1):
            id_node = obj.find("Object_ID")
            
            if id_node is None:
                id_element = ET.Element("Object_ID")
                id_element.text = str(g_id)
                obj.append(id_element)
                # obj.set("id", str(g_id))  # 添加 id 属性
                g_id += 1        

        formatted_xml = prettify_xml(root)

        # 保存格式化的 XML
        with open(dst_xml_path, "w", encoding="utf-8") as f:
            f.write(formatted_xml)
    print(g_id)
    return g_id

def main_hrsc(args):
    """默认 HRSC 自带 object id
    """
    return
    config_path = args.config

    config = Config.fromfile(config_path)
    load_modules(config.dataset)
    train_dataset = construct_dataset(config.dataset['train'])
    g_id = 1
    new_id = hrsc_id_objects(train_dataset, g_id)
    g_id = new_id


def dota_id_objects(dataset, g_id, dest: "str" = None, dataset_name="DOTA_v1"):
    """source 和 dest 目录可以相同
    """
    source: "str" = dataset.ann_folder
    data_root = osp.dirname(source)
    if dest is None:
        dest = osp.basename(source)
    os.makedirs(osp.join(data_root, dest), exist_ok=True)
    label_dir = source
    file_list = os.listdir(label_dir)
    files = sorted(file_list)
    for ori_label in tqdm.tqdm(files, total=len(file_list)):
        ori_label_path = osp.join(label_dir, ori_label)
        dst_label_path = osp.join(data_root, dest, ori_label)
        with open(ori_label_path) as f:
            lines = f.readlines()
        with open(dst_label_path, "w") as f:
            for line in lines:
                line = line.strip()
                if len(line) == 0: continue
                data = line.split(" ")
                if len(data) >= 10:
                    txt = " ".join(data[:10])
                    line = f"{txt} id:{g_id}"
                    g_id += 1
                f.write(line + "\n")
            f.flush()
    return g_id

def main_dota(args):
    config_path = args.config

    config = Config.fromfile(config_path)
    load_modules(config.dataset)
    train_dataset = construct_dataset(config.dataset['train'])
    g_id = 1
    new_id = dota_id_objects(train_dataset, g_id, 
        dest="labelTxt"
    )
    g_id = new_id
    print(f"Dota objects: {g_id}")

def main(args=None):
    parser = construct_parser()
    args = parser.parse_args(args)
    if args.dataset == "dota":
        main_dota(args)
    elif args.dataset == "dior":
        main_dior(args)        
    elif args.dataset == "hrsc":
        raise ValueError("HRSC dataset has object id by default.")
        main_hrsc(args)       
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 0:
        args = ["--dataset=dior", "--config=configs/dior_r/ds_dior_origin.py"]
    main(args)

