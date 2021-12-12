#!/usr/bin/python3
import numpy as np
import open3d as o3d
import cv2

with np.load('3Dcoordinates.npz') as data:
    points_3D = data['image_3d']

with np.load('stereoParams.npz') as data:
    intrinsics1 = data['intrinsics1']
    distortion1 = data['distortion1']
    intrinsics2 = data['intrinsics2']
    distortion2 = data['distortion2']
    R = data['R']
    T = data['T']
    E = data['E']
    F = data['F']

# Reading image
left = cv2.imread('#Lab5and6Images/left01.jpg')
undistort_left = cv2.undistort(left, intrinsics1, distortion1)

print(points_3D.shape)
p = points_3D.reshape(-1, 3)
u = np.array(undistort_left).reshape(-1, 3)


fp = []
fc = []
for i in range(p.shape[0]):
    if np.all(~np.isinf(p[i])) and np.all(~np.isnan(p[i])) and p[i][2] < 12 and p[i][2]>0:
        fp.append(p[i])
        fc.append(u[i]/255)
        # print(p[i].shape)
        # print(p[i])

# Create array of random points between [-1,1]
pcl = o3d.geometry.PointCloud()
# pcl.points = o3d.utility.Vector3dVector(np.random.rand(2500,3) * 2 - 1)
pcl.points = o3d.utility.Vector3dVector(fp)
pcl.colors = o3d.utility.Vector3dVector(fc)
#pcl.paint_uniform_color([0.0, 0.0, 0.0])

# Create axes mesh
Axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)

# shome meshes in view
o3d.visualization.draw_geometries([pcl , Axes])

