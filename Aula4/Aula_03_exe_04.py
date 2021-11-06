# Aula_03_ex_04.py
#
# Mean Filter

#import
import sys
import numpy as np
import cv2

def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1;
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)

# Read the image from argv
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
image = cv2.imread( "./lena.jpg", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./Lena_Ruido.png", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./DETI_Ruido.png", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./fce5noi3.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./fce5noi4.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./fce5noi6.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./sta2.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./sta2noi1.bmp", cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

cv2.namedWindow( "Original Image", cv2.WINDOW_AUTOSIZE )
cv2.imshow('Original Image', image)

# Median filter 3x3
imageGFilter3x3_1 = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
cv2.namedWindow( "Gaussian Filter 3x3 - 1 Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Gaussian Filter 3x3 - 1 Iter", imageGFilter3x3_1)
# Median filter 5x5
imageGFilter5x5_1 = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
cv2.namedWindow( "Gaussian Filter 5x5 - 1 Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Gaussian Filter 5x5 - 1 Iter", imageGFilter5x5_1)
# Median filter 7x7
imageGFilter7x7_1 = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_DEFAULT)
cv2.namedWindow( "Gaussian Filter 7x7 - 1 Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Gaussian Filter 7x7 - 1 Iter", imageGFilter7x7_1)
# Median filter 3x3 - 3 Iter
imageGFilter3x3_1 = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
imageGFilter3x3_2 = cv2.GaussianBlur(imageGFilter3x3_1, (3, 3), cv2.BORDER_DEFAULT)
imageGFilter3x3_3 = cv2.GaussianBlur(imageGFilter3x3_2, (3, 3), cv2.BORDER_DEFAULT)
cv2.namedWindow( "Gaussian Filter 3x3 - 3 Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Gaussian Filter 3x3 - 3 Iter", imageGFilter3x3_3)
cv2.waitKey(0)