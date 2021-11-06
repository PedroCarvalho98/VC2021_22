import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def compute_histogram(image,histSize, histRange):
	# Compute the histogram
	hist_item = cv2.calcHist([image], [0, 1, 2], None, [histSize], histRange)
	return hist_item

##########################################
# Drawing with openCV
# Create an image to display the histogram
def histogram2image(hist_item, histSize, histImageWidth, histImageHeight, color):

	histImage = np.zeros((histImageWidth, histImageHeight, 1), np.uint8)

	# Width of each histogram bar
	binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

	# Normalize values to [0, histImageHeight]
	cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)

	# Draw the bars of the nomrmalized histogram
	for i in range(histSize):
		cv2.rectangle(histImage, (i * binWidth, 0), ((i + 1) * binWidth, int(hist_item[i])), color, -1)

	# ATTENTION : Y coordinate upside down
	histImage = np.flipud(histImage)

	return histImage


# Read the image from argv
image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)


# Display the image
cv2.namedWindow("Original Image")
cv2.imshow("Original Image", image)

chans = cv2.split(image)
colors = ('b', 'g', 'r')

# Size
histSize = 256	 # from 0 to 255
# Intensity Range
histRange = [0, 256]
histImageWidth = 512
histImageHeight = 512
color = (256, 0, 0)

for (chan, color) in zip(chans, colors):
	hist_item = cv2.calcHist([chan], [0], None, histSize, histRange)
	histImage = histogram2image(hist_item, histSize, histImageWidth, histImageHeight, color)
	cv2.imshow('colorhist2', histImage)
	cv2.waitKey(0)
