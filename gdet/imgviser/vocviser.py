import xml.etree.ElementTree as ET
from .baseviser import Visualizer
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import cv2

class VocVisualizer(Visualizer) :

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse(self, filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                float(bbox.find("xmin").text),  #BFP_MARK
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
                float(bbox.find("angle").text),
            ]
            objects.append(obj_struct)

        return objects

    def vis(self, name):
        """
        name: str without suffix
        """
        
        file = self.imgdir + name + self.ImgExt
        image = io.imread(file)  # 读取图像
        instances = self.parse(self.labeldir + name + ".xml")
        for polyobj in instances:
            poly = polyobj["bbox"]
            poly = util.rotRecToPolygon(poly)
            poly = np.array(poly, dtype=np.int32)
            print(polyobj["bbox"], poly)
            poly = poly.reshape((-1, 2))
            s = len(poly)
            for i in range(s):
                image = cv2.line(image, tuple(poly[i]), tuple(poly[(i+1)%s]), (255,0,0), 2, lineType=cv2.LINE_AA)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(image)
        try:
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state('zoomed')    #最大化
        except : pass
        plt.show()
