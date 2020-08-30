
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
from colorsys import hsv_to_rgb

import matplotlib.pyplot as plt

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class YolactDetections:

    def __init__(self, shape, rois, masks, class_ids, scores):
        self.shape = shape
        self.rois = rois
        self.masks = masks
        self.class_ids = class_ids # stored as ints
        self.scores = scores

    def __len__(self):
        return self.masks.shape[2]

    @property
    def class_names(self):
        return [CLASS_NAMES[i] for i in self.class_ids]

   
    def get_background(self):
        bg_mask = np.logical_not(np.logical_or.reduce(self.masks, axis=2))
        return bg_mask

