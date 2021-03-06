#!/usr/bin/python3
import argparse
import timeit


import cv2
import numpy as np
import json


# /home/vinicius/Desktop/PSR_Git/psr_21-22/Parte05/images/atlas2000_e_atlasmv.png


# Create each trackbar function using globals
def onTrackbarminBH(minBH):
    global minimumBH
    minimumBH = minBH
    print("Selected threshold " + str(minBH) + " for limit B/H min")

def onTrackbarmaxBH(maxBH):
    global maximumBH
    maximumBH = maxBH
    print("Selected threshold " + str(maxBH) + " for limit B/H max")

def onTrackbarminGS(minGS):
    global minimumGS
    minimumGS = minGS
    print("Selected threshold " + str(minGS) + " for limit G/S min")

def onTrackbarmaxGS(maxGS):
    global maximumGS
    maximumGS = maxGS
    print("Selected threshold " + str(maxGS) + " for limit G/S max")

def onTrackbarminRV(minRV):
    global minimumRV
    minimumRV = minRV
    print("Selected threshold " + str(minRV) + " for limit R/V min")

def onTrackbarmaxRV(maxRV):
    global maximumRV
    maximumRV = maxRV
    print("Selected threshold " + str(maxRV) + " for limit R/V max")

def main():
    # Global variables
    global maximumBH, maximumGS, maximumRV, minimumBH, minimumGS, minimumRV

    # initial values for the globals
    minimumBH = 0
    minimumGS = 0
    minimumRV = 0
    maximumBH = 255
    maximumGS = 255
    maximumRV = 255

    window_name = 'Background'
    cv2.namedWindow(window_name)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image',
                        type=str,
                        required=True,
                        help='Full path to image that will be processed.')
    args = vars(parser.parse_args())

    # Create Trackbars
    cv2.createTrackbar('Min B/H', window_name, 0, 255, onTrackbarminBH)
    cv2.createTrackbar('Max B/H', window_name, 255, 255, onTrackbarmaxBH)
    cv2.createTrackbar('Min G/S', window_name, 0, 255, onTrackbarminGS)
    cv2.createTrackbar('Max G/S', window_name, 255, 255, onTrackbarmaxGS)
    cv2.createTrackbar('Min R/V', window_name, 0, 255, onTrackbarminRV)
    cv2.createTrackbar('Max R/V', window_name, 255, 255, onTrackbarmaxRV)

    while True:

        # Frame captured by the Video
        image = cv2.imread(args['image'], cv2.IMREAD_COLOR)
        image_original = image.copy()

        # Ranges dictionary used in trackbars
        ranges = {'limits': {'B/H': {'max': maximumBH, 'min': minimumBH},
                             'G/S': {'max': maximumGS, 'min': minimumGS},
                             'R/V': {'max': maximumRV, 'min': minimumRV}}}

        # Process image and creat a mask
        mins = np.array([ranges['limits']['B/H']['min'], ranges['limits']['G/S']['min'], ranges['limits']['R/V'][
            'min']])  # Converts the dictionary representation in np.array, which is the representation required by the inRange function
        maxs = np.array([ranges['limits']['B/H']['max'], ranges['limits']['G/S']['max'], ranges['limits']['R/V']['max']])
        mask = cv2.inRange(image, mins, maxs)

        # update image
        cv2.namedWindow(window_name)
        mask = cv2.resize(mask, (400, 400))
        cv2.imshow(window_name, mask)
        key = cv2.waitKey(1)

        cv2.namedWindow('original')
        image_original = cv2.resize(image_original, (400, 400))
        cv2.imshow('original',image_original)


        # Press q to quit the program and saves the directory
        if key == ord('q'):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('Writing dictionary ranges to file ' + file_name)
                json.dump(ranges, file_handle)
                print(ranges)
                break


if __name__ == "__main__":
    main()
