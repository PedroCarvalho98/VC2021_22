#!/usr/bin/python3

#------------- Autores------------------#

# Pedro Carvalho    | 84670
# Tiago Pinho       | 92938

#---------------------------------------#


import argparse
import cv2
import numpy as np
import json

# Create each trackbar function using globals
def onTrackbarminB(minB):
    global minimumBH
    minimumBH = minB
    print("Selected threshold " + str(minB) + " for limit B min")

def onTrackbarmaxB(maxB):
    global maximumBH
    maximumBH = maxB
    print("Selected threshold " + str(maxB) + " for limit B max")

def onTrackbarminG(minG):
    global minimumGS
    minimumGS = minG
    print("Selected threshold " + str(minG) + " for limit G min")

def onTrackbarmaxG(maxG):
    global maximumGS
    maximumGS = maxG
    print("Selected threshold " + str(maxG) + " for limit G max")

def onTrackbarminR(minR):
    global minimumRV
    minimumRV = minR
    print("Selected threshold " + str(minR) + " for limit R min")

def onTrackbarmaxR(maxR):
    global maximumRV
    maximumRV = maxR
    print("Selected threshold " + str(maxR) + " for limit R max")

def main():
    # Global variables
    global maximumB, maximumG, maximumR, minimumB, minimumG, minimumR

    # ---------------------------------------------------
    # Definition of Parser Arguments
    # ---------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-j',
                        '--json_JSON',
                        type=str,
                        required=True,
                        help='Full path to json file.')
    parser.add_argument('-i',
                        '--image',
                        type=str,
                        required=True,
                        help='Full path to image that will be processed.')

    args = vars(parser.parse_args())

    image = cv2.imread(args['image'], cv2.IMREAD_COLOR)

    # Open imported json
    lim = open(args['json_JSON'])
    ranges = json.load(lim)
    lim.close()

    # Convert dict. into np.arrays to define the minimum thresholds
    min_thresh = np.array([ranges['limits']['B/H']['min'],
                           ranges['limits']['G/S']['min'],
                           ranges['limits']['R/V']['min']])

    # Convert dict. into np.arrays to define the maximum thresholds
    max_thresh = np.array([ranges['limits']['B/H']['max'],
                           ranges['limits']['G/S']['max'],
                           ranges['limits']['R/V']['max']])

    # Create mask using the Json extracted Thresholds
    mask_segmented = cv2.inRange(image, min_thresh, max_thresh)

    res = cv2.bitwise_and(image, image, mask=mask_segmented)

    contours, hierarchy = cv2.findContours(
        mask_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Mask applied', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()