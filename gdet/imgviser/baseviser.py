"""1.2
"""
import multiprocessing as mp

import cv2

import numpy as np
import json
import os
import os.path as osp
from matplotlib import pyplot as plt

from fs.mathops import cal_line_length

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=1,
          color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    txtpos = (x, int(y + text_h + font_scale - 1))
    cv2.putText(img, text, txtpos, font, font_scale, color, font_thickness)

    return text_size


class Visualizer():
    ImgExt=".png"
    
    def __init__(self, basedir, labelDir="labelTxt") -> None:
        self.basedir = basedir
        self._imgdir = "images"
        self._labeldir = labelDir

    @property
    def imgdir(self):
        return osp.join(self.basedir, self._imgdir) + "/"

    @property
    def labeldir(self):
        if self._labeldir.startswith("/"):
            return self._labeldir
        return osp.join(self.basedir, self._labeldir) + "/"

    def vis(self, name): pass

