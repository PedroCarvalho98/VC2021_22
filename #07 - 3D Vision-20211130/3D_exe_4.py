#!/usr/bin/python3
import numpy as np
import open3d as o3d
import cv2

filt_office1 = o3d.io.read_point_cloud('Depth_Images/filt_office1.pcd')
filt_office2 = o3d.io.read_point_cloud('Depth_Images/filt_office2.pcd')

# Create axes mesh
Axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)

# shome meshes in view
o3d.visualization.draw_geometries([filt_office1, filt_office2, Axes])
# o3d.visualization.draw_geometries([pcl2 , Axes])



