# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import sys
import wsgi
import math
from matplotlib import pyplot as plt
from glob import glob
from os import rename
import pathlib



# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
	#help="path to input image")
#ap.add_argument("-y", "--yolo", required=True,
	#help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
	#help="minimum probability to filter weak detections")
#ap.add_argument("-t", "--threshold", type=float, default=0.3,
	#help="threshold when applying non-maxima suppression")
#args = vars(ap.parse_args())


folder='images/downloads/poodle/'
dest='images/downloads/crop/'

i=0
for fname in glob(folder+'*.*'):
	print(fname)
	ext = pathlib.Path(fname).suffix
	print(ext)
	num = pathlib.Path(fname).stem
	print(num)

	image = cv2.imread(fname)
	if image is None:
		continue
	#image = cv2.imread(args["image"])
	#final_list = wsgi.grabcutbg(image,5,5,5,5)
	final_list1 = wsgi.histcutbg(image,1,10,5,1)
	#final_list2 = wsgi.removebg(image,5,5,5,5)
	#final_list3 = wsgi.graycutbg(image,2,2,5,1)
	i=0
	for final_list in [ final_list1 ]:
		for im1 in final_list:
			print("showing {} {}".format(im1['key'],im1['label']) )
			final_img = im1['img']
			filename = "{}{}_crop_{}_{}_{}.png".format(dest,num,im1['label'],im1['key'],i)
			cv2.imwrite(filename, final_img)
		i = i + 1
	
	
	
	
	
	
	# Plot the final result
	# cv2.imread reads the image as BGR, but matplotlib uses RGB
	# BGR to RGB so we can plot the image with accurate colors
	#b, g, r = cv2.split(final_img)
	#final_img = cv2.merge([r, g, b])
	#plt.imshow(final_img)
	#plt.show()

	#cv2.imshow('image',final_img)
	#cv2.waitKey(0)