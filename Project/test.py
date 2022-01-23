#!/usr/bin/python3

from __future__ import print_function
import cv2
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import ROTATE_180
from cv2 import ROTATE_90_COUNTERCLOCKWISE


use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5

def main():
    
    global img
    global templ
    img = cv2.imread('./Imagens/Puzzle_X_completo_v3.jpeg', cv2.IMREAD_COLOR)
    templ = cv2.imread('./Colored/Colored2.jpg', cv2.IMREAD_COLOR)
    for l in [0, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE]:
        templ = cv2.rotate(templ, ROTATE_90_CLOCKWISE)
        
        cv2.namedWindow( image_window, cv2.WINDOW_AUTOSIZE )
        cv2.namedWindow( result_window, cv2.WINDOW_AUTOSIZE )
        
        
        trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
        cv2.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
        
        MatchingMethod(match_method)
        
    cv2.waitKey(0)
    return 0

def MatchingMethod(param):
    global match_method
    match_method = param
    
    img_display = img.copy()
    
    method_accepts_mask = (cv2.TM_SQDIFF == match_method or match_method == cv2.TM_CCORR_NORMED)
    result = cv2.matchTemplate(img, templ, match_method)
    
    
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    
    
    if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    
    cv2.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv2.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv2.imshow(image_window, img_display)
    cv2.imshow(result_window, result)
    cv2.imshow('peca', templ)
    cv2.waitKey(0)
    
    pass
if __name__ == "__main__":
    main()
