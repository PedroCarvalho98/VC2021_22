#!/usr/bin/python3

#------------- Autores------------------#

# Pedro Carvalho    | 84670
# Tiago Pinho       | 92938

#---------------------------------------#

import argparse
from turtle import width
from typing import final
import cv2
from cv2 import imshow
from cv2 import mean
from cv2 import waitKey
from cv2 import vconcat
from matplotlib import pyplot as plt, widgets
import numpy as np
from numpy.lib.function_base import copy
import os
import math

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
    # Which puzzle are we doing
    var = ''
    while(var != 'X' and var != 'Y' and var != 'Z' and var != 'W'):
        var = input('Qual é o puzzle? ')
        if var == 'X':
            filename_ordenado = "./Imagens/Puzzle_X_ordenado_v4.jpeg"
            filename_completo = "./Imagens/Puzzle_X_completo_v3.jpeg"
            max_k = 19
            wg, hg = 4, 5
        elif var == 'Y':
            filename_ordenado = "./Imagens/Puzzle_Y_ordenado.jpeg"
            filename_completo = "./Imagens/Puzzle_Y_completo.jpeg"
            max_k = 24
            wg, hg = 5, 5
        elif var == 'Z':
            filename_ordenado = "./Imagens/Puzzle_Z_ordenado.jpeg"
            filename_completo = "./Imagens/Puzzle_Z_completo.jpeg"
            max_k = 15
            wg, hg = 4, 4
        elif var == 'W':
            filename_ordenado = "./Imagens/Puzzle_W_ordenado.jpeg"
            filename_completo = "./Imagens/Puzzle_W_completo.jpeg"
            max_k = 11
            wg, hg = 4, 3
        else:
            print('Escolheu algo que nao existe')

    mascara = ''
    if not os.path.isfile('./Colored/Colored'+'0'+'_'+str(var)+'.jpg'):
        mascara = 'Sim'
    else:
        while (mascara != 'Sim' and mascara != 'Nao'):
            mascara = input('Deseja fazer a mascara (Sim/Nao): ')
            if mascara != 'Sim' and mascara != 'Nao':
                print('Enganou-se')

    if mascara == 'Sim':
        ## --- Retirar Background --- ##
        image = cv2.imread(filename_ordenado, cv2.IMREAD_COLOR)
        
        window_name = "Peças separadas"
        if var != 'X':
            image = cv2.resize(image, (800, 800))   # Temos demasiada resolução da câmara
        image_aux = image.copy()
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
        
        ## --- Afinação da máscara --- #

        eroded_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
        eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_DILATE, kernel)
        
        ## --- Segmentação e alinhamento das peças --- ##
        # Convert to graycsale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(img_gray,100,130)
        contours, hierarchy = cv2.findContours(~eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("img_gray", img_gray)
        cv2.imshow("edges", edges)
        cv2.waitKey(-1)
        contornos = []

        # Imagem com aplicação da máscara
        res = cv2.bitwise_and(image, image, mask=eroded_mask)
        background=cv2.bitwise_and(image, image, mask=~eroded_mask)
        res=cv2.bitwise_not(res)
        result=cv2.multiply(background,res)
        cv2.imshow("Com mascara", result)

        # Discartar áreas irrelevantes
        for i in contours:
            #print(cv2.contourArea(i))
            if cv2.contourArea(i) > 4000 and cv2.contourArea(i) <30000:
                contornos.append(i)

        for i in range(len(contornos)):
            cv2.drawContours(image_aux, contornos, i, (0,0,0), 2)
            # cv2.waitKey(200)
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
            #cv2.waitKey(200)
            pathBW = "./BW"
            pathColored = "./Colored"
            cv2.imwrite(os.path.join(pathBW , "BW"+str(i)+'_'+str(var)+".jpg"), cropped_bw)
            cv2.imwrite(os.path.join(pathColored , "Colored"+str(i)+'_'+ str(var)+".jpg"), cropped_color)

    cv2.waitKey(-1)

    # --- Corner Harris --- #
    # filename = './Colored/Colored'+'0'+'_'+str(var)+'.jpg'
    # img = cv2.imread(filename)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # dst = cv2.dilate(dst,None)
    # img[dst>0.1*dst.max()]=[0,0,255]
    # cv2.imshow('dst',img)
    # cv2.waitKey(-1)

    # # --- SIFT --- #
    # read images
    boas_pecas = []
    img7 = cv2.imread(filename_completo)
    img7 = cv2.resize(img7, (800, 800))   # Temos demasiada resolução da câmara 
    stop=False
    contflag=False
    ratio=0.6
    while stop != True:
        for k in range(0, max_k):
            img5=img7.copy()
            img6 = cv2.imread('./Colored/Colored'+str(k)+'_'+str(var)+'.jpg') 
            if len(boas_pecas)==(max_k+1):
                stop=True
                break
            for l in boas_pecas:
                if k == l[0]:
                    contflag=True
                    continue
            if contflag:
                contflag=False
                continue
            cv2.imshow('img5', img5)
            cv2.imshow('img6', img6)

            img1 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
            
            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 15)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)


            MIN_MATCH_COUNT = 10 
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

            else:
                print ("Not enough matches are found" )
                matchesMask = None
                continue

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = None)

            # # --- Matching --- #
            height,width,_=img5.shape
            Grid = [[0 for x in range(wg)] for y in range(hg)] 
            for j in range(hg):
                for i in range(wg):
                    Grid[j][i] = (int(width/(wg*2) + i*width/wg),int(height/(hg*2) + j*height/hg))
            
            point_list=[(int(point[0][0]),int(point[0][1])) for point in src_pts]
            good_points=[a*b for a,b in zip(point_list,matchesMask) if a*b!=0]
            good_points=[point for point in good_points if point!=()]
            if good_points==[]:
                print("No good quality points")
                continue
            #print(good_points)
            x_list=[x[0] for x in good_points]
            y_list=[y[1] for y in good_points]
            
            dst_pts_mean = (int(sum(x_list)/len(x_list)),int(sum(y_list)/len(y_list)))
            #print(dst_pts_mean)
            cv2.circle(img5, dst_pts_mean, 5, (0, 255, 0),-1)
            
            # --- Blue Points --- #
            for point in good_points:
                cv2.circle(img5, point, 5, (255, 0, 0),-1)

            mindist=None
            for j in range(hg):
                for i in range(wg):
                    distance=math.dist(Grid[j][i],dst_pts_mean)
                    if mindist==None or distance<mindist:
                        mindist=distance
                        closerpoint=Grid[j][i]
                        col=i
                        row=j
            minpoint=(int(min(x_list)),int(min(y_list)))
            maxpoint=(int(max(x_list)),int(max(y_list)))
            block_minpoint=(closerpoint[0]-int(width/(wg*2) ),closerpoint[1]-int(height/(hg * 2) ))
            block_maxpoint=(closerpoint[0]+int(width/(wg*2) ),closerpoint[1]+int(height/(hg * 2) ))
            cv2.circle(img5, closerpoint, 5, (0, 0, 255),-1)
            if math.dist(maxpoint,minpoint) > 400:
                cv2.rectangle(img5,minpoint,maxpoint,(0,255,255),5)
                NearGrid=[]
                TrueGrid=None
                for j in range(hg):
                    for i in range(wg):
                        NearGridCounter=0
                        for point in good_points:
                            mingrid=(Grid[j][i][0]-int(width/(wg*2) ),Grid[j][i][1]-int(height/(hg * 2) ))
                            maxgrid=(Grid[j][i][0]+int(width/(wg*2) ),Grid[j][i][1]+int(height/(hg * 2) ))
                            if mingrid[0] < point[0] < maxgrid[0] and mingrid[1] < point[1] < maxgrid[1]:
                                NearGridCounter+=1
                        NearGrid.append((Grid[j][i],NearGridCounter,(j,i)))
                
                for gridpoint in NearGrid:
                    if gridpoint[1]> 0.7 * len(good_points):
                        TrueGrid=gridpoint
                if TrueGrid != None:
                    print("Passed by Grid")
                    mingrid=(TrueGrid[0][0]-int(width/(wg*2) ),TrueGrid[0][1]-int(height/(hg * 2) ))
                    maxgrid=(TrueGrid[0][0]+int(width/(wg*2) ),TrueGrid[0][1]+int(height/(hg * 2) ))
                    cv2.rectangle(img7,mingrid,maxgrid,(0,0,0),-1)
                    boas_pecas.append((k, TrueGrid[0],TrueGrid[2]))
                    print((k, TrueGrid[0],TrueGrid[2]))
            else:
                cv2.rectangle(img5,minpoint,maxpoint,(255,255,0),5)
                img2w, img2h = img2.shape
                min_row = closerpoint[0] - math.floor(img2w/2)
                max_row = closerpoint[0] + math.ceil(img2w/2)
                min_col = closerpoint[1] - math.floor(img2h/2)
                max_col = closerpoint[1] + math.ceil(img2h/2)
                img_compare = img1[min_row:max_row, min_col:max_col]
                aux=cv2.subtract(img_compare,img2)
                n_zeros = np.count_nonzero(aux==0)
                if n_zeros/len(aux) > 0.8:
                    cv2.rectangle(img7,block_minpoint,block_maxpoint,(0,0,0),-1)
                    boas_pecas.append((k, closerpoint,(col,row)))

            cv2.imshow('img5', img5) 
            cv2.waitKey(100) 
        ratio+=0.05
        if ratio>0.8:
            break
        print("-----------------")
        for g in boas_pecas:
            print(g)

    cv2.imshow("img7",img7) 
    cv2.waitKey(-1)
    cv2.destroyAllWindows 
    
    # --- Resultado Final --- #
    final_pecas = [[None for i in range(wg)] for j in range(hg)]
    for g in boas_pecas:
        final_pecas[g[2][1]][g[2][0]]=g[0]
    print(final_pecas)
    
    blacksquare = cv2.imread('./Imagens/Black.png')
    blacksquare=cv2.resize(blacksquare, (int(width/(wg*2)),int(height/(hg * 2))))
    count_final = 0 
    for h in final_pecas:
        count = 0
        for n in h:
            if n == None:
                fig = blacksquare
            else:
                fig = cv2.imread('./Colored/Colored'+str(n)+'_'+str(var)+'.jpg')
                fig=cv2.resize(fig, (int(width/(wg*2)),int(height/(hg * 2))))
            if count == 0:
                im_v = fig
            else:
                im_v = np.concatenate((im_v, fig), axis=1)
            count+=1
        if count_final==0:
            final_final=im_v
        else:
            final_final = np.concatenate((final_final, im_v), axis=0)
        count_final+=1
    
    cv2.imshow('Final', final_final)
    cv2.waitKey()    
    cv2.destroyAllWindows   
if __name__ == "__main__":
    main()
