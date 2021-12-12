#!/usr/bin/python3

import numpy as np
import cv2
import glob

def mouse_handler ( event , x , y , flags , params ) :
    p = np.array([x,y])
    color = np.random.randint(0, 255, 3).tolist()
    if event == cv2.EVENT_LBUTTONDOWN:
        epilineR = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), params[1], params[2])
        epilineR = epilineR.reshape(-1, 3)[0]
        print(epilineR)
        x1 = 0
        y1 = round((-epilineR[0] * x1 - epilineR[2]) / epilineR[1])
        x2 = params[3].shape[1]
        y2 = round((-epilineR[0] * x2 - epilineR[2]) / epilineR[1])
        print((x1, y1), (x2, y2))
        cv2.line(params[0], (x1, y1), (x2, y2), color)
        cv2.imshow('mapped left', mapped_img_left)
        cv2.imshow('mapped right', mapped_img_right)

with np.load('stereoParams.npz') as data:
    intrinsics1 = data['intrinsics1']
    intrinsics2 = data['intrinsics2']
    distortion1 = data['distortion1']
    distortion2 = data['distortion2']
    R = data['R']
    T = data['T']
    E = data['E']
    F = data['F']

print ( " Intrinsics 1 : " )
print (intrinsics1)
print ( " Intrinsics 2 : " )
print (intrinsics2)
print ( " Distortion 1: " )
print ( distortion1)
print ( " Distortion 2: " )
print ( distortion2)
print ('R: ' + str(R))
print ('T: ' + str(T))
print ('E: ' + str(E))
print ('F: ' + str(F))

# image_left = cv2.imread('#Lab5and6Images/StereoL0.bmp')
image_left = cv2.imread('#Lab5and6Images/left01.jpg')
# image_right = cv2.imread('#Lab5and6Images/StereoR0.bmp')
image_right = cv2.imread('#Lab5and6Images/right01.jpg')

if len(image_right.shape) == 2:
    height, width = image_right.shape
    nchannels = 1;
else:
    height, width, nchannels = image_right.shape

und_left = cv2.undistort(image_left, intrinsics1, distortion1)
und_right = cv2.undistort(image_right, intrinsics2, distortion2)

# cv2.imshow('left', und_left)
# cv2.imshow('right', und_right)

# cv2.setMouseCallback('left', mouse_handler, [und_right, 1, F, und_left])
# cv2.setMouseCallback('right', mouse_handler,  [und_left, 2, F, und_right])

R1 = np.zeros(shape = (3, 3))
R2 = np.zeros(shape = (3, 3))
P1 = np.zeros(shape = (3, 4))
P2 = np.zeros(shape = (3, 4))
Q = np.zeros(shape = (4, 4))

cv2.stereoRectify(intrinsics1, distortion1, intrinsics2, distortion2, (width, height), R, T, R1, R2, P1, P2, Q, flags = cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

# Map computation
print('InitUndistortRectifyMap')
map1x, map1y = cv2.initUndistortRectifyMap(intrinsics1, distortion1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(intrinsics2, distortion2, R2, P2, (width, height), cv2.CV_32FC1)

mapped_img_left = cv2.remap(image_left, map1x, map1y, cv2.INTER_LINEAR)
mapped_img_right = cv2.remap(image_right, map2x, map2y, cv2.INTER_LINEAR)

cv2.imshow('mapped left', mapped_img_left)
cv2.imshow('mapped right', mapped_img_right)

cv2.setMouseCallback('mapped left', mouse_handler, [mapped_img_right, 1, F, und_left])
cv2.setMouseCallback('mapped right', mouse_handler,  [mapped_img_left, 2, F, und_right])

# for i in range(0,height,25):
#     color = np.random.randint(0, 255, 3).tolist()
#     cv2.line(mapped_img_left, (0, i), (width, i), color)
#     cv2.line(mapped_img_right, (0, i), (width, i), color)


# concat_mapped = np.concatenate((mapped_img_left, mapped_img_right), axis = 1)
# concat_und = np.concatenate((und_left, und_right), axis = 1)
# cv2.imshow('mapped', concat_mapped)
# cv2.imshow('undistort', concat_und)

cv2.waitKey(-1)
cv2.destroyAllWindows()


