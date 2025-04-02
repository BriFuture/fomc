import numpy as np
import torch
import logging
import tqdm
import os.path as osp
from gdet.fvlm.modules import clip_utils as clip #  import clip_fn, load
logger = logging.getLogger("fvlm.utils")
clip_vit_name_map = {
    "vit_b32": "ViT-B/32",
    "vit_b16": "ViT-B/16",
    "vit_l14": "ViT-L/14",
}
class EmbeddingCreator():
    def __init__(self):
        pass

    def get_categories(self, cfg):

        categories = []
        ### remove '-'
        text_feature_names = cfg.get("TEXT_FEATURE_NAMES", None)
        if text_feature_names is None:
            for cat in cfg.ALL_CLASSES:
                cat = cat.replace('-', ' ').strip()
                categories.append(cat)
        else:
            for cat in cfg.ALL_CLASSES:
                cat = text_feature_names[cat] ## map it
                cat = cat.replace('-', ' ').strip()
                categories.append(cat)
        print(f"EC: {categories}")
        return categories
    def get_categories_from_json(self, json_file):
        """from coco json file
        """
        categories = []
        ### remove '-'
        text_feature_names = cfg.get("TEXT_FEATURE_NAMES", None)
        if text_feature_names is None:
            for cat in cfg.ALL_CLASSES:
                cat = cat.replace('-', ' ').strip()
                categories.append(cat)
        else:
            for cat in cfg.ALL_CLASSES:
                cat = text_feature_names[cat] ## map it
                cat = cat.replace('-', ' ').strip()
                categories.append(cat)
        print(f"EC: {categories}")
        return categories          
          
    def create_embedding(self, args, categories, dst: "str"="", load_cache=False):
        """用 clip 进行 text embedding
        """
        if load_cache and osp.exists(dst):
            print("Load embediings from ", dst)
            return np.load(dst)
        print("Saving embediings in ", dst)
        model_name = args['model']
        embed_name = model_name.replace("resnet_", "RN")
        model_loc = args['model_loc']
        if not model_loc:
            model_loc = model_name
        ### map vit name
        if model_loc in clip_vit_name_map:
            model_loc = clip_vit_name_map[model_loc]
        clip_model, clip_transform = clip.load(model_loc)

        clip_model = clip_model.to(torch.float32)
        clip_text_fn = clip_model.encode_text

        class_clip_features = []
        logger.info('Computing custom category text embeddings.')

        num_classes = args['num_classes']  ### bsf.c contains background
        categories = categories[:num_classes - 1]
        print("Categories: ", categories)
        for cls_name in tqdm.tqdm(categories, total=len(categories)):
            cls_feat = clip.clip_fn(cls_name, clip_text_fn)
            # cls_feat = clip_text_fn(cls_name)
            class_clip_features.append(cls_feat)

        logger.info('Preparing input data.')
        # text_embeddings = text_embeddings[np.newaxis, Ellipsis]
        text_embeddings = np.stack(class_clip_features, axis=0) ## (10, 1024)

        embed_name = model_name.replace("resnet_", "r")
        empty_count = num_classes - len(categories) - 1
        text_embeddings = self.append_background(text_embeddings, embed_name, clip_text_fn, empty_count)
        if dst:
            np.save(dst, text_embeddings)
        print("Saved embedding shape: ", text_embeddings.shape, f" Empty Class Count: {empty_count}")
        return text_embeddings
    
    def append_background(self, text_embeddings, embed_name, clip_text_fn, empty_count):
        embed_path = (f'./data/{embed_name}_bg_empty_embed.npy')

        ### load bg and empty
        if osp.exists(embed_path):
            background_embedding, empty_embeddings = np.load(embed_path)
        else:
            background_embedding = clip.clip_fn("background", clip_text_fn)
            empty_embeddings = clip.clip_fn("empty", clip_text_fn)
            print("Using custom bg and empty embeding!")
            ### save it
            np.save(embed_path, (background_embedding, empty_embeddings))
            print(f"Save embed path: {embed_path}")

        background_embedding = background_embedding[np.newaxis, Ellipsis]

        empty_embeddings = empty_embeddings[np.newaxis, Ellipsis]
        if empty_count > 0:
            tile_empty_embeddings = np.tile(empty_embeddings, (empty_count, 1)) ### A[80, 1024]
            
            text_embeddings = np.concatenate(
                (background_embedding, text_embeddings, tile_empty_embeddings), axis=0
            ) ## (11, 1024)
        else:
            text_embeddings = np.concatenate(
                (background_embedding, text_embeddings), axis=0
            )
        return text_embeddings
    

def create_embedding(args, categories, dst: "str"="", load_cache=False):
    """用 clip 进行 text embedding
    """
    ec = EmbeddingCreator()
    text_embeddings = ec.create_embedding(args, categories=categories, dst=dst, load_cache=load_cache)
    return text_embeddings

def load_embeddings(categories, args):
    model_name = args['model']
    embed_name = model_name.replace("resnet_", "RN")
    # clip_text_fn = clip_utils.get_clip_text_fn(model_name)
    clip_model, clip_transform = clip.load(embed_name)
    clip_model = clip_model.to(torch.float32)
    clip_text_fn = clip_model.encode_text
    class_clip_features = []
    logger.info("Computing custom category text embeddings.")
    for cls_name in tqdm.tqdm(categories, total=len(categories)):
        cls_feat = clip.clip_fn(cls_name, clip_text_fn)
        cls_feat = cls_feat.reshape(1, -1)
        # cls_feat = clip_text_fn(cls_name)
        class_clip_features.append(cls_feat)

    logger.info("Preparing input data.")
    text_embeddings = np.concatenate(class_clip_features, axis=0)
    embed_name = model_name.replace("resnet_", "r")
    embed_path = f'./data/{embed_name}_bg_empty_embed.npy'
    background_embedding, empty_embeddings = np.load(embed_path)
    background_embedding = background_embedding[np.newaxis, Ellipsis]
    empty_embeddings = empty_embeddings[np.newaxis, Ellipsis]
    max_num_cls = args['max_num_classes']
    tile_empty_embeddings = np.tile(
        empty_embeddings, (max_num_cls - len(categories) - 1, 1)
    )
    # Concatenate 'background' and 'empty' embeddings.
    text_embeddings = np.concatenate(
        (background_embedding, text_embeddings, tile_empty_embeddings), axis=0
    )
    text_embeddings = text_embeddings[np.newaxis, Ellipsis]
    return text_embeddings