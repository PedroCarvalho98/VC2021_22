#!/usr/bin/python3

from __future__ import print_function
import cv2
from matplotlib import pyplot as plt, widgets


use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5

def main():
    boas_pecas = []
    contflag=False
    img7 = cv2.imread('./Imagens/Puzzle_X_completo_v3.jpeg') 
    for k in range(0, 19):
        if len(boas_pecas)==20:
            stop=True
            break
        for l in boas_pecas:
            if k == l[0]:
                contflag=True
                continue
        if contflag:
            contflag=False
            continue    
        img5=img7.copy()
        img6 = cv2.imread('./Colored/Colored'+str(k)+'.jpg') 
        img1 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance) 
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

        plt.imshow(img3),plt.show()

if __name__ == "__main__":
    main()
