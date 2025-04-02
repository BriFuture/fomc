import os, os.path as osp
import logging
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image

from gdet.registries import VISER

from bfcommon.utils import convert_tensor_as_npint8

logger = logging.getLogger("gdet.vis")
def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img

class BaseViser():
    def __init__(self, class_names: "list[str]"=None, full_class_names: "list[str]" = None, 
                class_colors=None, **kwargs):
        self.img_root = kwargs.get("img_root", "checkpoints/debug_pic")
        os.makedirs(self.img_root, exist_ok=True)
        # class_names (list[str]): Names of each classes.
        self.class_names = class_names
        self.full_class_names = full_class_names
        self.class_colors = class_colors
        # thickness (int): Thickness of lines.
        self.thickness = 1
        self.storage_mode = None
        # font_scale (float): Font scales of texts.
        self.font_scale=0.6

    def plot(self):
        pass
    def init(self):
        pass
    def shutdown(self):
        pass
    
    def sync(self):
        """bsf.c 用于异步时的同步数据
        """
        pass
    def clear(self):
        pass
    def get_im_loc(self, shape: "tuple", im_idx : "int" = 0, hint="x"):
        H, W = shape
        loc = f"{self.img_root}/{hint}{im_idx}_{H}_{W}.png"
        return loc

@VISER.register_module()
class PureImgViser(BaseViser):

    def init(self):
        self.img_root = "checkpoints/pure_imgs"
        os.makedirs(self.img_root, exist_ok=True)

    def plot(self, imgs: "torch.Tensor", out_file, im_idx = 0):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): Image Tensor [N, C, H, W].
            show (bool): Whether to show the image.            
            out_file (str or None): The filename to write the image.
        """
        image, m1, m2 = convert_tensor_as_npint8(imgs[im_idx])
        if out_file is not None:
            img_loc = osp.join(self.img_root, out_file) # Test only
        else:
            out_file = ""
            img_loc = None
        # img = imread(img)

        if img_loc is not None:
            cv2.imwrite(image, img_loc)

    def clear(self):
        pass

    def mix_heatmap(self, imgs: "torch.Tensor", features: "torch.Tensor", scale=None, hmratio=0.5, 
            indicator="mix", id=0, im_idx=0, grid=False):
        """features: T[N, C, H, W] or T[C, H, W]
        使用 save_bimg_with_gtbbox 
        """
        if features.ndim == 4:
            feat = features[id]
        else:
            feat = features
        if scale is None:
            im_w = imgs.shape[-1]
            ft_w = feat.shape[-1]
            scale = int(im_w / ft_w)
        heatmap = self.feature_to_heatmap(feat, scale, grid)
        if heatmap is None:
            print("heatmap invalid")
            return
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        src_size = heatmap.shape
        rawim, m1, m2 = convert_tensor_as_npint8(imgs[im_idx])
        # rawim = cv2.imread(rawim_loc) # H W C
        superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
        im_file = f"{indicator}{id}_{src_size[0]}_{src_size[1]}.png"
        im_loc = osp.join(self.img_root, im_file)
        print(im_loc)
        cv2.imwrite(im_loc, superimposed_img)

    @staticmethod
    def feature_to_heatmap(features: "torch.Tensor", scale=8, near=False) -> "Image.Image":
        """ features CHW
        """
        assert features.ndim <= 3
        if features.ndim == 3:
            heatmap = torch.sum(features, dim=0)  # 尺度大小， 如torch.Size([1,45,45])
            size = features.shape[1:]  # 原图尺寸大小
        else:
            heatmap = features  # 尺度大小， 如torch.Size([1,45,45])
            size = features.shape  # 原图尺寸大小
        src_size = (size[1] * scale, size[0] * scale) # H W
        heatmap = PureImgViser.norm_heatmap(heatmap, src_size, near)
        print(f"src_size: {src_size[1]}_{src_size[0]}")
        return heatmap, src_size

    @staticmethod
    def norm_heatmap(heatmap: "torch.Tensor", src_size, near=False, cm=cv2.COLORMAP_JET, norm_type="min-max"):
        if type(heatmap) is torch.Tensor:
            heatmap = heatmap.detach().cpu().numpy()
        if norm_type == "min-max":
            max_value = np.max(heatmap)
            min_value = np.min(heatmap)
            diff_value = (max_value - min_value)
            if diff_value > 1e-3 or abs(max_value - 0) > 1e-3:
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
            else:
                heatmap *= 0
            print(f"max: {max_value:.2f} {min_value:.2f}")
        else:
            min_value = -1
            max_value = 1
            heatmap = (heatmap - min_value) / (max_value - min_value) * 255
            
        heatmap = heatmap.astype(np.uint8)# .transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
        if near:
            heatmap = cv2.resize(heatmap, src_size, interpolation=cv2.INTER_NEAREST)  # 重整图片到原尺寸
        else:
            heatmap = cv2.resize(heatmap, src_size, interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
        if cm is not None:
            heatmap = cv2.applyColorMap(heatmap, cm) # H W C
        else:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        return heatmap
    
class TensorViser(BaseViser):
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    def save_tensor(self, xim : "torch.Tensor", im_idx : "int" =0):
        """xim: T[3, h, w]"""
        arr, m1, m2 = convert_tensor_as_npint8(xim)
        im = Image.fromarray(arr)

        H, W = xim.shape[-2:]
        loc = self.get_im_loc((H, W), im_idx)
        im.save(loc)
        print(f"{loc} max: {m1:.2f} {m2:.2f}")
        return im, loc

    def save_np_as_img(self, xim : "np.ndarray", im_idx : "int" =0):
        """xim: T[h, w, 3]"""
        if xim.dtype != np.uint8:
            xim = imdenormalize(xim, **self.img_norm_cfg)
            xim = xim.astype(np.uint8)
        im = Image.fromarray(xim)

        H, W = xim.shape[:2]
        loc = self.get_im_loc((H, W), im_idx)
        im.save(loc)
        print(f"{loc}")
        return im, loc

    def save_bimg_tensor(self, xims: "torch.Tensor"):
        """xim: T[B, 3, h, w]"""
        for i, xim in enumerate(xims):
            self.save_tensor(xim, i)

    def save_heatmap(self, features: "torch.Tensor", scale=8, near=False) -> "Image.Image":
        """ features CHW
        """
        heatmap, src_size = PureImgViser.feature_to_heatmap(features, scale, near)
        # 保存热力图
        im = Image.fromarray(heatmap)
        im.save(f"data/images/xh_{src_size[1]}_{src_size[0]}.png")
        return heatmap

    def mix_heatmap(self, features: "torch.Tensor", scale=8, hmratio=0.5, indicator="mix", id=0, grid=False):
        """features: T[N, C, H, W] or T[C, H, W]
        使用 save_bimg_with_gtbbox 
        """
        if features.ndim == 4:
            feat = features[id]
        else:
            feat = features
        heatmap = self.save_heatmap(feat, scale, grid)
        if heatmap is None:
            print("heatmap invalid")
            return
        src_size = heatmap.shape
        rawim = f"{self.img_root}/x{id}_{src_size[0]}_{src_size[1]}.png"
        if not osp.exists(rawim):
            print("Not exist", rawim)
            return
        rawim = cv2.imread(rawim) # H W C
        superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
        loc = f"{self.img_root}/{indicator}{id}_{src_size[0]}_{src_size[1]}.png"
        print(loc)
        cv2.imwrite(loc, superimposed_img)

    def mix_heatmap_separate(self, features: "torch.Tensor", scale=8, hmratio=0.5, indicator="mix"):
        """features: T[N, C, H, W] or T[C, H, W]
        使用 save_bimg_with_gtbbox 
        """
        if features.ndim == 4:
            feat = features[0]
        else:
            feat = features
        src_size = feat.shape[-2:]
        id = 0
        size = (src_size[0] * scale, src_size[1] * scale)
        
        rawim_loc = f"{self.img_root}/x{id}_{size[0]}_{size[1]}.png"
        if not osp.exists(rawim_loc):
            print("Not exist", rawim_loc)
            return
        rawim = cv2.imread(rawim_loc) # H W C
        os.makedirs(f"{self.img_root}/seperate", exist_ok=True)
        for fidx in range(feat.shape[0]):
            sf = feat[fidx]
            heatmap = self.save_heatmap(sf, scale)
            dst_img = f"{self.img_root}/seperate/{indicator}-{fidx}_{size[0]}_{size[1]}.png"
            if heatmap is None:
                print("heatmap invalid")
                if osp.exists(dst_img):
                    os.remove(dst_img)
                continue
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            
            # print(rawim.shape)
            superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
            cv2.imwrite(dst_img, superimposed_img)

    # done
import torch.nn.functional as F
from tqdm import tqdm 

@VISER.register_module()
class SimViser(PureImgViser):
    def plot_hm(self, ximgs: "torch.Tensor", feat: "torch.Tensor", idx1: "int", 
                im_idx = 0, skip_self=True, hmratio = 0.8):
        """feat: neck output feature
        """
        if type(idx1) is torch.Tensor:
            idx1 = idx1.item()
        width = feat.shape[-1]
        scale = 1024 // width
        iy1, ix1 = divmod(idx1, width)
        if iy1 > width or ix1 > width:
            return None, None
        print("ix, iy: ", ix1, iy1, " <---> ", ix1 * scale, iy1 * scale,)
        src_size = (1024, 1024)
        t0  = feat[im_idx, :, iy1, ix1]
        heatmap = np.zeros((width, width))
        for iy2 in tqdm(range(width)):
            for ix2 in range(width):
                if iy2 == iy1 and ix2 == ix1 and skip_self:
                    print(f"Skip Self: {skip_self}")
                    continue
                rt0 = feat[im_idx, :, iy2, ix2]
                s = F.cosine_similarity(t0, rt0, -1)
                heatmap[iy2, ix2] = s

        norm_hm = self.norm_heatmap(heatmap, src_size, near=True, cm=None, norm_type="normal")
        rawim, m1, m2 = convert_tensor_as_npint8(ximgs[im_idx])
        superimposed_img = norm_hm * hmratio + rawim * (1-hmratio)
        im_file = f"{im_idx}_{src_size[0]}_{width}.png"
        im_loc = osp.join(self.img_root, im_file)
        print(im_loc)
        cv2.imwrite(im_loc, superimposed_img)
        im_file = f"hm_{im_idx}_{src_size[0]}_{width}.png"
        im_loc = osp.join(self.img_root, im_file)
        cv2.imwrite(im_loc, norm_hm)
        return heatmap, norm_hm

## test    
def calc_sim(x, idx1, idx2):
    if type(idx1) is torch.Tensor:
        idx1 = idx1.item()
    scale = 8
    width = 1024 // scale
    iy1, ix1 = divmod(idx1, width)
    if type(idx2) is torch.Tensor:
        idx2 = idx2.item()
    iy2, ix2 = divmod(idx2, width)
    print(ix1, iy1, " <---> ",  ix2, iy2)
    print(ix1 * scale, iy1 * scale, " <---> ", ix2 * scale, iy2 * scale)
    feat = x[0]
    t0  = feat[0, :, iy1, ix1]
    rt0 = feat[0, :, iy2, ix2]
    s = F.cosine_similarity(t0, rt0, -1)
    return s

def calc_sim_sim(x, ref_x, idx):
    if type(idx) is torch.Tensor:
        idx = idx.item()
    ix, iy = divmod(idx, 128)
    print(ix, iy)
    t0 = x[0][0, :, ix, iy]
    rt0 = ref_x[0][0, :, ix, iy]
    s = F.cosine_similarity(t0, rt0, -1)
    return s

