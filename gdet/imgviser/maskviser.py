from .baseviser import Visualizer
from skimage import io
import numpy as np
import cv2
import multiprocessing as mp
import tqdm
from matplotlib import pyplot as plt
import os, os.path as osp
from fs.core.bbox.transforms_rotated import poly_to_rotated_box_single, obb2quadbox_np_single
from .dotaviser import DotaVisualizer
from PIL import Image, ImageDraw

class MaskVisualizer(DotaVisualizer):
    def __init__(self, basedir, args) -> None:
        super().__init__(basedir, args)
        self.maskit = True
        self.display_poly_num = 3
    def listdir(self):
        spec_image = self.args['spec_image']
        assert spec_image is not None
        return super().listdir()

    def output(self, outdir, single_thread=True):
        return super().output(outdir, False)
    
    def do_mask(self, image: "np.ndarray", mask_polygons: "list"):
        # Create a black image as a mask (with the same dimensions as the original image)
        mask = np.zeros_like(image)
        # Apply Gaussian blur to the masked area of the image
        mask_polys = []
        for polyobj in mask_polygons:
            poly = polyobj["poly"]
            poly = self.convert_bbox_as_poly(poly)

            mask_polys.append(poly)
        mask_polys = np.asarray(mask_polys)
        cv2.fillPoly(mask, mask_polys, (255, 255, 255))
    
        # blur_image = np.where(mask != 0, image, 255)
        blur_image = image

        blur_image = cv2.GaussianBlur(blur_image, (151, 151), 150, )
        blur_poly_image = np.where(mask != 0, blur_image, image)
        return blur_poly_image

    def _output(self, imgfile):
        name, ext = osp.splitext(imgfile)

        labelloc = self.labeldir + name + ".txt"
        # print(labelloc)
        if not osp.exists(labelloc): 
            print("Label Loc not found: ", labelloc)
            return

        polygons = self.parse_poly(labelloc) 
        if len(polygons) == 0: return
        spec_class = self.args["spec_class"]
        contain_spec_class = False
        image = cv2.imread(osp.join(self.imgdir, imgfile))  # 读取图像
        
        unmask_polygon_ids = []
        mask_polygon_ids = []
        for pi, polyobj in enumerate(polygons):
            # color = (0, 0, 0)
            name = polyobj['name']
            name = self.CLASS_SHORT_NAME.get(name, "")
            if len(unmask_polygon_ids) < self.display_poly_num :
                unmask_polygon_ids.append(pi)
            else:
                mask_polygon_ids.append(pi)

        draw_polygons = [polygons[ind] for ind in unmask_polygon_ids]
        mask_polygons = [polygons[ind] for ind in mask_polygon_ids]
        if self.maskit:
            image = self.do_mask(image, mask_polygons)
        else:
            image = image

        drawed = True
        for pi, polyobj in enumerate(draw_polygons):
            # color = (0, 0, 0)
            cls_name = polyobj['name']
            name = self.CLASS_SHORT_NAME.get(cls_name, "PL")
            color = self.CLASS_COLOR[name]

            poly = polyobj["poly"]
            poly = self.convert_bbox_as_poly(poly)

            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1], color, self._line_width, lineType=cv2.LINE_AA)
            image = cv2.line(image, poly[3], poly[0], color, self._line_width, lineType=cv2.LINE_AA)
            drawed = True
            if spec_class is not None and cls_name == spec_class:
                contain_spec_class = True

            if "id" in polyobj and self._show_id:
                oid = polyobj["id"]
                image = cv2.putText(image, f"{oid}-{cls_name}", poly[3], cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=0.5)
        

        if drawed:
            if self.maskit:
                imgname, ext = osp.splitext(imgfile)
                imgfile = f"{imgname}_mask{ext}"
            
            loc = osp.join(self._outdir, imgfile)
            print(loc)
            cv2.imwrite(loc, image)