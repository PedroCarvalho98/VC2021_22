#!/usr/bin/python3

#------------- Autores------------------#

# Pedro Carvalho    | 84670
# Tiago Pinho       | 92938

#---------------------------------------#


import argparse
import cv2
import numpy as np

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
        print(self.listcolors)
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
        print(B,G,R)
        params.addtolist(B,G,R)
        Bmax,Gmax,Rmax,Bmin,Gmin,Rmin = params.maskvalues()
        print(Bmax,Gmax,Rmax,Bmin,Gmin,Rmin)
        minval=np.array([Bmin,Gmin,Rmin])
        maxval=np.array([Bmax,Gmax,Rmax])
        mask=cv2.inRange(image, minval, maxval)
        res = cv2.bitwise_and(image, image, mask=~mask)
        cv2.imshow("Mouse", res)

def main():

    ## --- Retirar Background --- ##
    image = cv2.imread("./Imagens/Puzzle_X_ordenado_v2.jpeg", cv2.IMREAD_COLOR)
    window_name = 'Peças separadas'
    image = cv2.resize(image, (800, 800))   # Temos demasiada resolução da câmara
    # mask = cv2.inRange(image, (0,80,0), (210,170,90))
    # res = cv2.bitwise_and(image, image, mask=~mask)
    masker=maskmaker(image)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    # cv2.imshow("mask", mask)
    cv2.imshow("Mouse",image)
    cv2.setMouseCallback("Mouse", mouse_handler,masker)
    cv2.waitKey(-1)
    Bmax,Gmax,Rmax,Bmin,Gmin,Rmin=masker.maskvalues()
    minval=np.array([Bmin,Gmin,Rmin])
    maxval=np.array([Bmax,Gmax,Rmax])
    mask = cv2.inRange(image, minval, maxval)
    kernel = np.ones((5,5), np.uint8)
    # Using cv2.erode() method
    eroded_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    # eroded_mask=cv2.morphologyEx(eroded_mask,cv2.MORPH_CLOSE, kernel)

    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_DILATE, kernel)
    kernel = np.ones((11, 11), np.uint8)
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)
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
