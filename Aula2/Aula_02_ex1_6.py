# Aula_02_ex1_6_py


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


cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )  

black_white = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   

cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", black_white)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )  