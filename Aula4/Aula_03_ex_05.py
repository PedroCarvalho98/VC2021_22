# Aula_03_ex_05.py
#
# Sobel Operator
#
# Paulo Dias - 09/2021

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

# Sobel Operator 3 x 3 - X
imageSobel3x3_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)

cv2.namedWindow( "Sobel 3 x 3 - X", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Sobel 3 x 3 - X", imageSobel3x3_X )
image8bits_X = np.uint8( np.absolute(imageSobel3x3_X) )
cv2.imshow( "8 bits - Sobel 3 x 3 - X", image8bits_X )

# Sobel Operator 3 x 3 - Y
imageSobel3x3_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1, 3)

cv2.namedWindow( "Sobel 3 x 3 - Y", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Sobel 3 x 3 - Y", imageSobel3x3_Y )
image8bits_Y = np.uint8( np.absolute(imageSobel3x3_Y) )
cv2.imshow( "8 bits - Sobel 3 x 3 - Y", image8bits_Y )

result3x3 = (imageSobel3x3_X**2 + imageSobel3x3_Y**2)**0.5
result8bits3x3 = np.uint8(result3x3)
cv2.imshow( "Sobel 3 x 3 - Result", result3x3)
cv2.imshow( "8 bits - Sobel 3 x 3 - Result", result8bits3x3)

# Sobel Operator 5 x 5 - X
imageSobel5x5_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, 5)

cv2.namedWindow( "Sobel 5 x 5 - X", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Sobel 5 x 5 - X", imageSobel5x5_X )
image8bits_X_5x5 = np.uint8( np.absolute(imageSobel5x5_X) )
cv2.imshow( "8 bits - Sobel 5 x 5 - X", image8bits_X_5x5 )

# Sobel Operator 5 x 5 - Y
imageSobel5x5_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1, 5)

cv2.namedWindow( "Sobel 5 x 5 - Y", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Sobel 5 x 5 - Y", imageSobel5x5_Y )
image8bits_Y_5x5 = np.uint8( np.absolute(imageSobel5x5_Y) )
cv2.imshow( "8 bits - Sobel 5 x 5 - Y", image8bits_Y_5x5 )

result5x5 = (imageSobel5x5_X**2 + imageSobel5x5_Y**2)**0.5
result8bits5x5 = np.uint8(result5x5)
cv2.imshow( "Sobel 5 x 5 - Result", result3x3)
cv2.imshow( "8 bits - Sobel 5 x 5 - Result", result8bits5x5)
cv2.waitKey(0)


