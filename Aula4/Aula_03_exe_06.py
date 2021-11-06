# Aula_03_ex_06.py

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
# image = cv2.imread( "./wdg2.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./cln1.bmp", cv2.IMREAD_GRAYSCALE );
# image = cv2.imread( "./Bikesgray.jpg", cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

cv2.imshow('Original Image', image)

# Canny Operator 3 x 3 - X
edges3x3_X = cv2.Canny(image, 75, 100, 3)

cv2.namedWindow( "Canny 3 x 3 - X", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Canny 3 x 3 - X", edges3x3_X )

# Canny Operator 3 x 3 - Y
edges3x3_Y = cv2.Canny(image, 75, 100, 3)

cv2.namedWindow( "Canny 3 x 3 - Y", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Canny 3 x 3 - Y", edges3x3_Y )


result3x3 = (edges3x3_X**2 + edges3x3_Y**2)**0.5
cv2.imshow( "Canny 3 x 3 - Result", result3x3)


cv2.waitKey(0)
