
import os.path as osp
import numpy as np
from .dotaviser import DotaVisualizer

def parse_nwpu_poly(fileloc):
    objects = []
    with open(fileloc) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            line = line.split(' ')
            assert len(line) == 6

            obj = {}
            obj['name'] = line[4]
            # obj['id'] = line[4]
            poly = list(map(int, line[:4]))
            poly = np.asarray([poly[0], poly[1], poly[0], poly[3], 
                poly[2], poly[3], poly[2], poly[1],])
            obj['poly'] = poly.reshape(-1, 2)
            obj['filename'] = fileloc
            obj['difficult'] = 0
            objects.append(obj)
    return objects

class NwpuVisualizer(DotaVisualizer):
    def __init__(self, basedir, labelDir="labelTxt") -> None:
        super().__init__(basedir, labelDir)
        # self.parse_poly = parse_nwpu_poly