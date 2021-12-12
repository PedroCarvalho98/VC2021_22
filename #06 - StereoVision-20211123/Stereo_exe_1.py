#!/usr/bin/python3

import numpy as np
import cv2
import glob

# Board Size (Temos de mudar caso queiramos usar o bmp para 10, 7)
board_h = 9
board_w = 6


# # Arrays to store object points and image points from all the images_left.
# objpoints = [] # 3d point in real world space
# left_points = [] # 2d points in image plane.


def FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    # If found, display image with corners
    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(50)

    return ret, corners


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images_left and images_right.
objpoints = []  # 3d point in real world space
left_points = []  # 2d points in image plane with left images.
right_points = []  # 2d points in image plane with right images.

# Read images_left
# images_left = sorted(glob.glob('#Lab5and6Images/left*.jpg'))
images_left = sorted(glob.glob('#Lab5and6Images/StereoL*.bmp'))
# images_right = sorted(glob.glob('#Lab5and6Images/right*.jpg'))
images_right = sorted(glob.glob('#Lab5and6Images/StereoR*.bmp'))

height = []
width = []
channels = []

for left_name in images_left:
    left_img = cv2.imread(left_name)
    # print(left_name)
    ret_left, corners_left = FindAndDisplayChessboard(left_img)
    height, width, channels = left_img.shape
    if ret_left == True:
        objpoints.append(objp)
        left_points.append(corners_left)

for right_name in images_right:
    right_img = cv2.imread(right_name)
    # print(right_name)
    ret_right, corners_right = FindAndDisplayChessboard(right_img)
    if ret_right == True:
        right_points.append(corners_right)

_, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_points,
                                                                                        right_points, None, None,
                                                                                        None, None, (width, height),
                                                                                        flags=cv2.CALIB_SAME_FOCAL_LENGTH)

print(" Intrinsics 1 : ")
print(intrinsics1)
print(" Intrinsics 2 : ")
print(intrinsics2)
print(" Distortion 1: ")
print(distortion1)
print(" Distortion 2: ")
print(distortion2)
print('R: ' + str(R))
print('T: ' + str(T))
print('E: ' + str(E))
print('F: ' + str(F))
np.savez('stereoParams.npz', intrinsics1=intrinsics1, distortion1=distortion1, intrinsics2=intrinsics2,
         distortion2=distortion2, R=R, T=T, E=E, F=F)

cv2.waitKey(-1)
cv2.destroyAllWindows()
