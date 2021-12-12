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
        cv2.imshow('left', und_left)
        cv2.imshow('right', und_right)

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

cv2.imshow('left', und_left)
cv2.imshow('right', und_right)

cv2.setMouseCallback('left', mouse_handler, [und_right, 1, F, und_left])
cv2.setMouseCallback('right', mouse_handler,  [und_left, 2, F, und_right])


cv2.waitKey(-1)
cv2.destroyAllWindows()
