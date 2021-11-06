import numpy as np
import cv2
import sys

image = cv2.imread( sys.argv[1], cv2.IMREAD_UNCHANGED );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics

if len(image.shape) < 3:
	height, width = image.shape
	print('gray')
	Type = 'Gray'
	for i in range(0, width, 20):
		cv2.line(image, (i, 0), (i , height), (255))
	for j in range(0, height, 20):
		cv2.line(image, (0, j), (width, j), (255))

elif len(image.shape) == 3:
	height, width, aux = image.shape
	print('RGB')
	Type = 'RGB'
	for i in range(0, width, 20):
		cv2.line(image, (i, 0), (i, height), (128,128,128))
	for j in range(0, height, 20):
		cv2.line(image, (0, j), (width, j), (128, 128, 128))
else:
	print('others')
	Type = 'Others'



# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )

cv2.imwrite('Teste.jpg', image)

