# Aula_02_ex1_5_py


#import
import numpy as np
import cv2
import sys

# Read the image
image = cv2.imread( sys.argv[1], cv2.IMREAD_UNCHANGED );

def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left click")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow( "Window", image)

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)


cv2.namedWindow( "Wndow", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Window", image)

cv2.setMouseCallback("Window", mouse_handler)

# Show the image


# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Window" )     