#!/usr/bin/env python
# coding: utf-8



import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# In[48]:


ext = np.array([[0.000528046,  -0.999968,  0.00801468,  0.0390392],
[-0.0381385,-0.00802898,-0.99924,-0.0438104],
[0.999272,0.000221977,-0.0381415,-0.063029]])
intr = np.array([[591.13,0,320.019],        
        [0,578.745,260.225],
        [0,0,1]])


# # LDLS Demo
# 
# This notebook demonstrates how to use LDLS to perform instance segmentation of a LiDAR point cloud. This demo uses Frame 571 from the KITTI object detection dataset.

# ## Setup
# 
# Import LiDAR segmentation modules:

# In[6]:


import numpy as np
from pathlib import Path
import skimage

from lidar_segmentation.detections import MaskRCNNDetections
from lidar_segmentation.segmentation import LidarSegmentation
from lidar_segmentation.kitti_utils import load_kitti_lidar_data, load_kitti_object_calib,KittiProjection
from lidar_segmentation.plotting import plot_segmentation_result
from lidar_segmentation.utils import load_image
from mask_rcnn.mask_rcnn import MaskRCNNDetector
import torch
import torch.backends.cudnn as cudnn
import gc
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


# In[149]:


import open3d as o3d
import plotly.io as pio
#input_num = 636
scene = 2
from PIL import Image
projection = KittiProjection.load_livox(intr,ext)

lidarseg = LidarSegmentation(projection)

print("ARGG ",sys.argv[1])
arg=int(sys.argv[1])-1
for input_num in range(arg*100,(arg*100)+100,10):
	print("RUNNING ",arg, "  --  ",arg+100)
	im3 = Image.open("/home/pave/vision_ws/LDLS/lidar_image_data/scene{}/image/image_{}.png".format(scene,format(input_num,"03d")))


	pcd = o3d.io.read_point_cloud("/home/pave/vision_ws/LDLS/lidar_image_data/scene{}/lidar/lidar_{}.pcd".format(scene,format(input_num,"03d")))
	# path2 = "/home/pave/Livox/catkin_camera_lidar_calibration/src/livox_camera_lidar_calibration/data/pcdFiles/15.pcd"
	# pcd = o3d.io.read_point_cloud(path2)
	np_points = np.asarray(pcd.points)

	sample = np.arange(len(np_points))
	np.random.shuffle(sample)
	len(sample)
	sample = np.arange(len(np_points))
	np.random.shuffle(sample)
	sampled_points = np_points[sample[:len(sample)//4]]


	detector=MaskRCNNDetector()
	detection = detector.detect(np.array(im3))


	import time


	st = time.time()
	# results = lidarseg.run(lidar, detection, max_iters=20,save_all=False)
	results = lidarseg.run(sampled_points, detection, max_iters=20,save_all=False)

	print(1/(time.time() - st), "FPS")



	pio.write_image(plot_segmentation_result(results, label_type='class'),"yolact_results/" + "scene" + str(scene) +  "/image_" +  format(input_num,"03d") + ".png",format='png')

	gc.collect()
	del pcd,im3



