# ***********************************************************************************
# Name:           viewcloud.py
# Revision:
# Date:           30-10-2019
# Author:         Paulo Dias
# Comments:       Viewcloud
#
# images         left1.jpg->left19.jpg
# Revision:
# Libraries:    Python 3.7.5 + openCV 4.1.0
# ***********************************************************************************
import numpy as np
import open3d as o3d

# Create array of random points between [-1,1]
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(np.random.rand(2500,3) * 2 - 1)
#pcl.paint_uniform_color([0.0, 0.0, 0.0])

# Create axes mesh
Axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)

# shome meshes in view
o3d.visualization.draw_geometries([pcl , Axes])



