# Aula_02_ex1_4_py


#import
import numpy as np
import cv2
import sys

# Read the image
image1 = cv2.imread( sys.argv[1], cv2.IMREAD_UNCHANGED );
image2 = cv2.imread( sys.argv[2], cv2.IMREAD_UNCHANGED );


if  np.shape(image1) == ():
	# Failed Reading
	print("Image1 file could not be open")
	exit(-1)

if  np.shape(image2) == ():
 	# Failed Reading
 	print("Image2 file could not be open")
 	exit(-1)    

# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image1)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )        

cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image2)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )        
                
diff = cv2.subtract(image1,image2); 

cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", diff)

# Wait
cv2.waitKey( 0 );

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )  