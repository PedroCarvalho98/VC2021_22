# Aula_02_ex1_3_py


#import
import numpy as np
import cv2
import sys

# Read the image
image = cv2.imread( sys.argv[1], cv2.IMREAD_UNCHANGED );


if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics
height, width = image.shape
image2 = image.copy()
# image2 = [image2 < 128]
for j in range(height):
    for i in range(width):
        if (image2[i, j] < 128):
            image2[i, j] = 0


# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image )

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )

cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image2 )

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
