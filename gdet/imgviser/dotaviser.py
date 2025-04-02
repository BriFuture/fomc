from .baseviser import Visualizer
from skimage import io
import numpy as np
import cv2
import multiprocessing as mp
import tqdm
from matplotlib import pyplot as plt
import os, os.path as osp
from fs.core.bbox.transforms_rotated import poly_to_rotated_box_single, obb2quadbox_np_single

class DotaVisualizer(Visualizer):
    def __init__(self, basedir, args) -> None:
        
        super().__init__(basedir, args["labeldir"])
        import fs.dota.dota_utils as util
        from fs.dota.meta import CLASS_COLOR, CLASS_SHORT_NAME
        self.parse_poly = util.parse_dota_poly
        self.CLASS_COLOR = CLASS_COLOR
        self.CLASS_SHORT_NAME = CLASS_SHORT_NAME
        self.args = args
        self._show_id    = args['show_id']
        self._line_width = args['line_width']
        self._max_img_count = args['max_image']
        self._box_type = args['box_type']
        assert self._box_type in [0, 1, 2]
        print("Box type", self._box_type)
    
    def vis(self, name):
        polygons = self.parse_poly(self.labeldir + name + ".txt")
        image = io.imread(self.imgdir + name + self.ImgExt)  # 读取图像

        for pi, polyobj in enumerate(polygons):
            poly = polyobj["poly"]
            poly = np.array(poly, dtype=np.int32)
            color = (255, (pi * 30) % 255, 0)
            if self._box_type == 0:
                pass
            elif self._box_type == 1:
                # print(poly.reshape(-1))
                pass
            elif self._box_type == 2:
                poly = poly.reshape(-1)
                xmin = np.min(poly[0::2])
                xmax = np.max(poly[0::2])
                ymin = np.min(poly[1::2])
                ymax = np.max(poly[1::2])
                poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                

            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1],color, 1, lineType=cv2.LINE_AA)

            image = cv2.line(image, poly[3], poly[0],color, 3, lineType=cv2.LINE_AA)

        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(image)
        try:
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state('zoomed')    #最大化
        except : pass
        plt.show()
    
    def convert_bbox_as_poly(self, poly: "np.ndarray"):
        poly = np.array(poly, dtype=np.int32)
        if self._box_type == 0:
            pass
        elif self._box_type == 1:
            # print(poly.reshape(-1))
            poly = poly.reshape(-1)
            rbb  = poly_to_rotated_box_single(poly)
            poly = obb2quadbox_np_single(rbb)
            poly = poly.reshape(-1, 2).astype(np.int32)
            
        elif self._box_type == 2:
            poly = poly.reshape(-1)
            xmin = np.min(poly[0::2])
            xmax = np.max(poly[0::2])
            ymin = np.min(poly[1::2])
            ymax = np.max(poly[1::2])
            poly = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], dtype=np.int32)

        return poly
    
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
        drawed = False
        for pi, polyobj in enumerate(polygons):
            # color = (0, 0, 0)
            if int(polyobj["difficult"]) == 0: 
                name = polyobj['name']
                name = self.CLASS_SHORT_NAME.get(polyobj["name"], "PL")
                color = self.CLASS_COLOR[name]
            elif int(polyobj["difficult"]) == 1: 
                # color = (30, 30, 225)
                # continue
                name = polyobj['name']
                name = self.CLASS_SHORT_NAME.get(polyobj["name"], "PL")
                color = self.CLASS_COLOR[name]
            elif int(polyobj["difficult"]) == 2:
                color = (0, 0, 225)
                continue

            poly = polyobj["poly"]
            poly = self.convert_bbox_as_poly(poly)

            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1], color, self._line_width, lineType=cv2.LINE_AA)
            image = cv2.line(image, poly[3], poly[0], color, self._line_width, lineType=cv2.LINE_AA)
            drawed = True
            cls_name = polyobj["name"]
            if spec_class is not None and cls_name == spec_class:
                contain_spec_class = True
            if "id" in polyobj and self._show_id:
                oid = polyobj["id"]
                # print("---> ", oid, poly)
                image = cv2.putText(image, f"{oid}-{cls_name}", poly[3], cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=0.5)
        
        if spec_class and not contain_spec_class:
            drawed = False

        if drawed:
            if self._box_type == 2:
                imgname, ext = osp.splitext(imgfile)
                imgfile = f"{imgname}_hbb{ext}"
            loc = osp.join(self._outdir, imgfile)
            cv2.imwrite(loc, image)
            # print(loc)
            
    def listdir(self):        
        list_files = list(os.listdir(self.imgdir))
        
        files = []
        spec_image = self.args['spec_image']
        for name in list_files:
            if "hf" in name: continue
            if "vf" in name: continue
            if "df" in name: continue
            if "_unblur"  in name: continue
            if "cutmix"  in name: continue
            if spec_image is not None and spec_image not in name:
                continue
            
            files.append(name)
        return files
    

    
    def output(self, outdir, single_thread = True):
        spec_class = self.args["spec_class"]
        if spec_class is not None:
            outdir = osp.join(self.basedir, f"{outdir}_{spec_class}")
            self._outdir = outdir
        else:
            outdir = osp.join(self.basedir, outdir)
            self._outdir = outdir

        os.makedirs(outdir, exist_ok=True)

        files = self.listdir()

        if self._max_img_count > 0:
            files = files[:self._max_img_count]
            print(files[:min(self._max_img_count, 10)])

        print("Filtering raw, unmask", self.imgdir, len(files), single_thread)
        if single_thread:
            for imgfile in tqdm.tqdm(files, total=len(files)):
                self._output(imgfile)
        else:
            pool = mp.Pool(8)
            r = list(tqdm.tqdm(pool.imap(self._output, files), 
                        total=len(files), desc=f"Output bbox: {outdir}") )
