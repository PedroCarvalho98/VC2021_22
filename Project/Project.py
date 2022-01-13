#!/usr/bin/python3

#------------- Autores------------------#

# Pedro Carvalho    | 84670
# Tiago Pinho       | 92938

#---------------------------------------#

import argparse
import cv2
import numpy as np
from numpy.lib.function_base import copy
import os

class maskmaker():

    def __init__(self,img):
        self.img=img
        self.listcolors=[]
    def addtolist(self,B,G,R):
        self.listcolors.append([B,G,R])
    def getimage(self):
        return self.img
    def getlist(self):
        return self.listcolors
    def maskvalues(self):
        Bmax = 0
        Bmin = 255
        Gmax = 0
        Gmin = 255
        Rmax = 0
        Rmin = 255
        for i in self.listcolors:
            if i[0] > Bmax:
                Bmax=i[0]
            if i[0] < Bmin:
                Bmin=i[0]
            if i[1] > Gmax:
                Gmax = i[1]
            if i[1] < Gmin:
                Gmin = i[1]
            if i[2] > Rmax:
                Rmax = i[2]
            if i[2] < Rmin:
                Rmin = i[2]

        return np.float32(Bmax+5),np.float32(Gmax+5),np.float32(Rmax+5),np.float32(Bmin-5),np.float32(Gmin-5),np.float32(Rmin-5)



def mouse_handler(event, x, y,flags, params):
    if event== cv2.EVENT_LBUTTONDOWN:
        image=params.getimage()
        B,G,R=image[y][x]
        params.addtolist(B,G,R)
        Bmax,Gmax,Rmax,Bmin,Gmin,Rmin = params.maskvalues()
        minval=np.array([Bmin,Gmin,Rmin])
        maxval=np.array([Bmax,Gmax,Rmax])
        mask=cv2.inRange(image, minval, maxval)
        res = cv2.bitwise_and(image, image, mask=~mask)
        cv2.imshow("Mouse", res)

def main():

    ## --- Retirar Background --- ##
    image = cv2.imread("./Imagens/Puzzle_X_ordenado_v2.jpeg", cv2.IMREAD_COLOR)
    image_aux = image.copy()
    window_name = "Peças separadas"
    image = cv2.resize(image, (800, 800))   # Temos demasiada resolução da câmara
    image_aux = cv2.resize(image_aux, (800, 800))
    masker=maskmaker(image)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.imshow("Mouse",image)
    cv2.setMouseCallback("Mouse", mouse_handler,masker)
    cv2.waitKey(-1)
    
    ## --- Aplicação da máscara --- #
    
    Bmax,Gmax,Rmax,Bmin,Gmin,Rmin=masker.maskvalues()
    minval=np.array([Bmin,Gmin,Rmin])
    maxval=np.array([Bmax,Gmax,Rmax])
    mask = cv2.inRange(image, minval, maxval)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    ## -- Afinação da máscara --- #

    eroded_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_DILATE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)
    
    ## --- Segmentação e alinhamento das peças --- ##
    # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img_gray,100,130)
    contours, hierarchy = cv2.findContours(~eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("img_gray", img_gray)
    cv2.imshow("edges", edges)
    contornos = []

    # Imagem com aplicação da máscara
    res = cv2.bitwise_and(image, image, mask=eroded_mask)
    background=cv2.bitwise_and(image, image, mask=~eroded_mask)
    res=cv2.bitwise_not(res)
    result=cv2.multiply(background,res)
    cv2.imshow("Com mascara", result)

    # Discartar áreas irrelevantes
    for i in contours:
        if cv2.contourArea(i) > 4000:
            contornos.append(i)

    for i in range(len(contornos)):
        cv2.drawContours(image_aux, contornos, i, (0,0,0), 2)
        cv2.waitKey(200)
        cv2.imshow("Contours", image_aux)
    
    # Isolar cada peça
    list_of_figures_cropped_black_white = []
    list_of_figures_cropped_color = []
    for i in range(len(contornos)):
        x, y = [], []
        for contour_line in contornos[i]:
            for contour in contour_line:
                x.append(contour[0])
                y.append(contour[1])

        x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

        cropped_bw = result[y1:y2, x1:x2]
        cropped_color = background[y1:y2, x1:x2]
        list_of_figures_cropped_black_white.append(cropped_bw)
        list_of_figures_cropped_color.append(cropped_color)
        cv2.imshow("Cropped bw", cropped_bw)
        cv2.imshow("Cropped color", cropped_color)
        cv2.waitKey(200)
        pathBW = "./BW"
        pathColored = "./Colored"
        cv2.imwrite(os.path.join(pathBW , "BW"+str(i)+".jpg"), cropped_bw)
        cv2.imwrite(os.path.join(pathColored , "Colored"+str(i)+".jpg"), cropped_color)


    cv2.waitKey()
    cv2.destroyAllWindows
if __name__ == "__main__":
    main()
