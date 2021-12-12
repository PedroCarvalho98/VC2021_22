#!/usr/bin/python3

import numpy as np
import cv2
import glob

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

und_left = cv2.undistort(image_left, intrinsics1, distortion1)
und_right = cv2.undistort(image_right, intrinsics2, distortion2)

concat_left = np.concatenate((image_left, und_left), axis = 1)
concat_right = np.concatenate((image_right, und_right), axis = 1)
cv2.imshow('left', concat_left)
cv2.imshow('right', concat_right)
cv2.waitKey(-1)
cv2.destroyAllWindows()
