import numpy as np
import cv2
import sys

image = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED);
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)


cv2.namedWindow( "Original Image", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Original Image", image)

cv2.namedWindow( "Thresh1", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Thresh1", thresh1)
cv2.namedWindow( "Thresh2", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Thresh2", thresh2)
cv2.namedWindow( "Thresh3", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Thresh3", thresh3)
cv2.namedWindow( "Thresh4", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Thresh4", thresh4)
cv2.namedWindow( "Thresh5", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Thresh5", thresh5)
cv2.waitKey( 0 );
cv2.destroyWindow( "Original Image" )
cv2.destroyWindow( "Thresh1" )
cv2.destroyWindow( "Thresh2" )
cv2.destroyWindow( "Thresh3" )
cv2.destroyWindow( "Thresh4" )
cv2.destroyWindow( "Thresh5" )
