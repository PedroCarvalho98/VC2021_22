#!/usr/bin/python3

#------------- Autores------------------#

# Pedro Carvalho    | 84670
# Tiago Pinho       | 92938

#---------------------------------------#


import argparse
import cv2
import numpy as np

def main():

    ## --- Retirar Background --- ##
    image = cv2.imread("./Imagens/Puzzle_X_ordenado_v2.jpeg", cv2.IMREAD_COLOR)
    window_name = 'Peças separadas'
    image = cv2.resize(image, (800, 800))   # Temos demasiada resolução da câmara
    mask = cv2.inRange(image, (0,80,0), (130,150,90))
    res = cv2.bitwise_and(image, image, mask=~mask)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.imshow("mask", mask)

    kernel = np.ones((13, 13), np.uint8)
    # Using cv2.erode() method 
    eroded_mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    eroded_mask=cv2.morphologyEx(eroded_mask,cv2.MORPH_OPEN, kernel)
    # cv2.imshow("sem fundo", res)
    
    ## --- Segmentação e alinhamento das peças --- ##
    # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    
    edges = cv2.Canny(img_gray,100,130)
    contours, hierarchy = cv2.findContours(~eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("img_gray", img_gray)
    cv2.imshow("edges", edges)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imshow('Contours', image)


    cv2.waitKey()
    cv2.destroyAllWindows
if __name__ == "__main__":
    main()
