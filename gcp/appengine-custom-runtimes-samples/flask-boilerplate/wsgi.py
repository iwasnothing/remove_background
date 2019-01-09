from flask import Flask
from flask import make_response
import numpy as np
import time
import cv2
from datetime import datetime
from google.cloud import storage
from flask_cors import CORS
from flask import jsonify
import math
import json
from flask import request
import itertools

app = Flask(__name__)
CORS(app)



@app.route("/")
def hello():
    return "Docker Hello World!"

@app.route('/now')
def shownow():
	return str(datetime.now())

def distance(x1,y1,x2,y2):
	return math.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )

def index2bin(x):
	nb=8
	return math.floor(x / int(256/nb+1))
def get_corner_color(img,l=3):
	(row,column,channel) = img.shape
	dx = 30
	#l = 10
	result = []
	images = [img[:dx,:dx], img[:dx,column-dx:], img[row-dx:,:dx], img[row-dx:,column-dx:]]
	for im in images:
		h = gethist(im)
		#print("histogram")
		#print(h)
		colors = topcolor(h,l)
		#print("top colors")
		#print(colors)
		result.extend(colors)

	return result
	
def iscorner(i,j,row,column):
	corner = False
	dx = 2
	dy = 2
	#print("corner is {} {}".format(dx,dy))
	if(i<=dx and j<=dy):
		corner = True
	elif(i<=dx and j>=column-dy):
		corner = True
	elif(i>=row-dx and j <= dy):
		corner = True
	elif(i>=row-dx and j >=column-dy):
		corner = True
	else:
		corner = False
		
	return corner

def gethistcnt(image):
# floor(x / int(256/3+1))
	nb = 8
	return cv2.calcHist([image], [0, 1, 2],
		None, [nb, nb, nb], [0, 256, 0, 256, 0, 256])

def cnt2pct(hist):
	hist1d = hist.flatten()
	total = sum(hist1d)
	hist = ( hist * 100 ) / total
	return hist
	
def gethist(image):
# floor(x / int(256/3+1))
	nb = 8
	hist = cv2.calcHist([image], [0, 1, 2],
		None, [nb, nb, nb], [0, 256, 0, 256, 0, 256])
	hist1d = hist.flatten()
	total = sum(hist1d)
	hist = ( hist * 100 ) / total
	return hist

def topcolor(hist, threshold):
	result = []
	mylist = []
	for index, x in np.ndenumerate(hist):
		jdoc = {'key': index, 'value': x}
		mylist.append(jdoc)
	dsort = sorted(mylist,key=lambda k: k['value'], reverse=True)
	upbound = max(1,min(threshold,len(dsort)))
	for jdoc in dsort[:upbound]:
		result.append(jdoc['key'])
		
	return result

def mywater(img,marker):
	marked = cv2.watershed(img, marker)
	# Make the background black, and what we want to keep white
	marked[marked == 1] = 0
	marked[marked > 1] = 255

	# Use a kernel to dilate the image, to not lose any detail on the outline
	# I used a kernel of 3x3 pixels
	kernel = np.ones((3,3),np.uint8)
	dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations = 1)

	# Plot again to check whether the dilation is according to our needs
	# If not, repeat by using a smaller/bigger kernel, or more/less iterations
	#plt.imshow(dilation, cmap='gray')
	#plt.show()

	# Now apply the mask we created on the initial image
	final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))
	return final_img
def bgtransparent(final_img):
	tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
	_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
	b, g, r = cv2.split(final_img)
	rgba = [b,g,r, alpha]
	dst = cv2.merge(rgba,4)
	return dst
	
def removebg(image,dynrange1=25,dynrange2=25,dynrange3=25,dynrange4=25):
	# load the COCO class labels our YOLO model was trained on
	#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	print("[INFO] loading YOLO {} {} {} {}".format(dynrange1,dynrange2,dynrange3,dynrange4) )
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
		
	# derive the paths to the YOLO weights and model configuration
	#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	print("[INFO] loaded YOLO label")
	# derive the paths to the YOLO weights and model configuration
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	#image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	print("image size = {} {}".format(H,W))
	ratio = max(W/640,H/480)
	#image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)
		
	final_list=[]
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			print("processing {} {}".format(LABELS[classIDs[i]],i)) 
			one_img = {"key": i, "label": LABELS[classIDs[i]]}
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			# Rectange values: start x, start y, width, height
			rectangle = (x, y, w, h)
			print("size is {} {}".format(w,h))
			if ( w*h < 100 ) :
				continue
			crop_img = image[y:y+h, x:x+w]
			print(crop_img.shape[:2])
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)

			# show the output image
			#imS = cv2.resize(crop_img, (640, 480))
			(smallH, smallW) = crop_img.shape[:2]
			if ( smallW*smallH < 100 ) :
				continue
			print("image size = {} {}".format(smallH, smallW))
			ratio = max(smallW/640,smallH/480/2)*2
			img = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )
			print("canny edge detection")
			#img = crop_img
			#dynrange = 25
			edgestart = 100
			edgeend = edgestart + dynrange1
			print(crop_img.shape[:2])
			edges = cv2.Canny(img,edgestart,edgeend)
			#cv2.imshow('edge',edges)
			#cv2.waitKey(0)		
		
			(row,column,channel) = img.shape
			print("remove background")
			marker = np.zeros_like(img[:,:,0]).astype(np.int32)
			
			pctx1 = np.zeros(column).astype(np.double)
			pctx99 = np.zeros(column).astype(np.double)
			pcty1 = np.zeros(row).astype(np.double)
			pcty99 = np.zeros(row).astype(np.double)
			#print(pcty1.shape)
			lowbound = dynrange1
			highbound = 100 - lowbound
			for i in range(row):
				temp_list = []
				#print(i)
				for j in range(column):
					if(edges[i][j] > 0):
						temp_list.append(j)
				#print(temp_list)
				if( len(temp_list) > 0):
					pcty1[i] = np.percentile(temp_list,lowbound)
					pcty99[i] = np.percentile(temp_list,highbound)
				else:
					pcty1[i] = 0
					pcty99[i] = column
			for j in range(column):			
				temp_list = []
				for i in range(row):
					if(edges[i][j] > 0):
						temp_list.append(i)
				if( len(temp_list) > 0):
					pctx1[j] = np.percentile(temp_list,lowbound)
					pctx99[j] = np.percentile(temp_list,highbound)		
				else:
					pctx1[j] = 0
					pctx99[j] = row

			objx = []
			objy = []
			
			for i in range(row):	
				for j in range(column):
					if(edges[i][j] > 0):
						objy.append(i)
						objx.append(j)
			#print(len(objx))
			#print(len(objy))	
			if (len(objx) == 0 or len(objy) == 0):
				continue
			a = np.array(objx)
			b = np.array(objy)
			mid=50
			fixrange = dynrange2
			dynrange = fixrange
			pct1 = max(0,mid - dynrange)
			pct2 = min(100,mid + dynrange)
			pct3 = min(dynrange,pct1)
			pct4 = min(max(100-dynrange,pct2),100)
			#print(len(a))
			#print(len(b))
			px1 = np.percentile(a, pct1)
			py1 = np.percentile(b, pct1)
			px2 = np.percentile(a, pct2)
			py2 = np.percentile(b, pct2)
			px3 = np.percentile(a, pct3)
			py3 = np.percentile(b, pct3)
			px4 = np.percentile(a, pct4)
			py4 = np.percentile(b, pct4)
			midx = np.percentile(a, 50)
			midy = np.percentile(b, 50)

			dx = np.std(a)
			dy = np.std(b)
			dv = 0.1*dynrange3*math.sqrt(dx*dx+dy*dy)
			lx = max(a)
			ly = max(b)


			print("obj is {},{},{},{}".format(px1,px2,py1,py2) )

			for i in range(row):	
				for j in range(column):
					#if ( i > pctx1[j] and i < pctx99[j] and j > pcty1[i] and j < pcty99[i]):
					if( distance(j,i,midx,midy) <= dv):
						marker[i][j] = 255
					elif (j > px1 and j < px2 and i > py1 and i < py2):
						marker[i][j] = 99
					elif ( (j < px3 or j > px4) and (i < py3 or i > py4) ):
						marker[i][j] = 1
					else:
						marker[i][j] = 0
					
			# close color
			colorbuck = 10
			currentRGB = np.zeros(3, dtype=int)
			centerRGB = np.zeros( (colorbuck,colorbuck,colorbuck) , dtype=int )
			cornerRGB = np.zeros( (colorbuck,colorbuck,colorbuck) , dtype=int )
			centerCNT = 0
			cornerCNT = 0
			for i in range(row):	
				for j in range(column):
					#if ( i > pctx1[j] and i < pctx99[j] and j > pcty1[i] and j < pcty99[i]):
					if (j > px1 and j < px2 and i > py1 and i < py2):
						for k in range(3):
							currentRGB[k] = int(img[i,j,k] / int(260/colorbuck))
						centerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] = centerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] + 1
						centerCNT = centerCNT + 1
					elif ( (j < px3 or j > px4) and (i < py3 or i > py4) and distance(j,i,midx,midy) > dv  ):
						for k in range(3):
							currentRGB[k] = int(img[i,j,k] / int(260/colorbuck))
						cornerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] = cornerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] + 1
						cornerCNT = cornerCNT + 1
			print("cnt: {} {}".format(centerCNT,cornerCNT) )
			for i in range(colorbuck):
				for j in range(colorbuck):
					for k in range(colorbuck):

						centerRGB[i,j,k] = int(centerRGB[i,j,k]*100 / (centerCNT + 1) )
						cornerRGB[i,j,k] = int(cornerRGB[i,j,k]*100 / (cornerCNT + 1) )

			colorgap = dynrange4
			for i in range(row):	
				for j in range(column):
					currentRGB = np.zeros(3, dtype=int)
					for k in range(3):
						currentRGB[k] = int(img[i,j,k] / int(260/colorbuck))
					#if ( abs(img[i,j,0]-centerRGB[0]) < colorgap and abs(img[i,j,1]-centerRGB[1]) < colorgap and abs(img[i,j,2]-centerRGB[2]) < colorgap ):
					if( centerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] > colorgap ):
						#print("center")
						marker[i][j] = 255
					elif ( currentRGB[0] < 3 and currentRGB[1] < 3 and currentRGB[2] < 3  and distance(j,i,midx,midy) < 3*dv ):
						marker[i][j] = 255
					elif ( cornerRGB[currentRGB[0],currentRGB[1],currentRGB[2]] > colorgap and distance(j,i,midx,midy) > dv ):
						#print("background")
						marker[i][j] = 1

			#print(marker[0])
			marked = cv2.watershed(img, marker)

			# Plot this one. If it does what we want, proceed;
			# otherwise edit your markers and repeat
			#plt.imshow(marked, cmap='gray')
			#plt.show()



			# Make the background black, and what we want to keep white
			#marked[marked == 1] = 0
			#marked[marked > 1] = 255

			# Use a kernel to dilate the image, to not lose any detail on the outline
			# I used a kernel of 3x3 pixels
			kernel = np.ones((3,3),np.uint8)
			dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations = 1)

			# Plot again to check whether the dilation is according to our needs
			# If not, repeat by using a smaller/bigger kernel, or more/less iterations
			#plt.imshow(dilation, cmap='gray')
			#plt.show()

			# Now apply the mask we created on the initial image
			final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))


			tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
			_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
			b, g, r = cv2.split(final_img)
			rgba = [b,g,r, alpha]
			dst = cv2.merge(rgba,4)
			one_img['img'] = dst
			one_img['cropim'] = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )

			final_list.append(one_img)
	return final_list

def grabcutbg(image,dynrange1=25,dynrange2=25,dynrange3=25,dynrange4=25):
	# load the COCO class labels our YOLO model was trained on
	#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	print("[INFO] loading YOLO {} {} {} {}".format(dynrange1,dynrange2,dynrange3,dynrange4) )
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
		
	# derive the paths to the YOLO weights and model configuration
	#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	print("[INFO] loaded YOLO label")
	# derive the paths to the YOLO weights and model configuration
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	#image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	dynrange1 = max(1,dynrange1)
	ratio = max(max(W/1024,H/768)/dynrange1,1)
	print("image size = {} {} {}".format(H,W,ratio))
	image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	#image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	h0 = gethist(image)
	h0cnt = gethistcnt(image)
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)
		
	final_list=[]
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			print("processing {} {}".format(LABELS[classIDs[i]],i)) 
			one_img = {"key": i, "label": LABELS[classIDs[i]]}
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			# Rectange values: start x, start y, width, height
			rectangle = (x, y, w, h)
			print("size is {} {}".format(w,h))
			if ( w*h < 100 ) :
				continue
			crop_img = image[y:y+h, x:x+w]
			print(crop_img.shape[:2])
			h1 = gethist(crop_img)
			h1cnt = gethistcnt(crop_img)
			forethreshold = dynrange2
			backthreshold = dynrange4
			forecolor = topcolor(h1-h0,forethreshold)
			backcolor = topcolor(cnt2pct(h0-h1),backthreshold)
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)

			# show the output image
			#imS = cv2.resize(crop_img, (640, 480))
			(smallH, smallW) = crop_img.shape[:2]
			if ( smallW*smallH < 100 ) :
				continue
			print("image size = {} {}".format(smallH, smallW))
			ratio = max(smallW/640,smallH/480)
			
			mask = np.zeros(image.shape[:2],np.uint8)
			
			#  check if color bin = top colr mask = 255, else 1, 


			bgdModel = np.zeros((1,65),np.float64)
			fgdModel = np.zeros((1,65),np.float64)

			rect = rectangle
			print("start grab cut")
			cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
			mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
			final_img = image*mask2[:,:,np.newaxis]
			(smallH, smallW) = final_img.shape[:2]
			ratio = max(smallW/640,smallH/480)
			final_img = cv2.resize(final_img, (int(smallW/ratio), int(smallH/ratio)) )
			print("done grab cut")
			
			tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
			_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
			b, g, r = cv2.split(final_img)
			rgba = [b,g,r, alpha]
			dst = cv2.merge(rgba,4)
			one_img['img'] = dst
			one_img['cropim'] = crop_img
			final_list.append(one_img)
	return final_list

def histcutbg(image,dynrange1=25,dynrange2=25,dynrange3=25,dynrange4=25):
	# load the COCO class labels our YOLO model was trained on
	#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	print("[INFO] loading YOLO {} {} {} {}".format(dynrange1,dynrange2,dynrange3,dynrange4) )
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
		
	# derive the paths to the YOLO weights and model configuration
	#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	print("[INFO] loaded YOLO label")
	# derive the paths to the YOLO weights and model configuration
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	#image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	if ( H*W > 800 ):
		dynrange1 = max(1,dynrange1)
		ratio = max(max(W/1024,H/768)/dynrange1,1)
		print("image size = {} {} {} {}".format(H,W,dynrange1,ratio))
		image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
		(H, W) = image.shape[:2]
	#image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	#h0 = gethist(image)
	#h0cnt = gethistcnt(image)
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)
		
	final_list=[]
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			print("processing {} {}".format(LABELS[classIDs[i]],i)) 
			one_img = {"key": i, "label": LABELS[classIDs[i]]}
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			# Rectange values: start x, start y, width, height
			rectangle = (x, y, w, h)
			print("size is {} {}".format(w,h))
			if ( w*h < 100 ) :
				continue
			crop_img = image[y:y+h ,x:x+w]
			# show the output image
			#imS = cv2.resize(crop_img, (640, 480))
			(smallH, smallW) = crop_img.shape[:2]
			if ( smallW*smallH < 100 ) :
				continue
			print("crop image size = {} {}".format(smallH, smallW))
			ratio = smallW/150
			
			#crop_img = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )
			crop_img = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )
			(smallH, smallW) = crop_img.shape[:2]
			if ( smallW*smallH < 100 ) :
				continue
			print("crop image resize = {} {}".format(smallH, smallW))
			#img = crop_img
			h0 = gethist(crop_img)
			h0cnt = gethistcnt(crop_img)
			#crop_center = image[ x+int(w/3):x+int(w*2/3),y+int(h/3):y+int(h*2/3)]
			base = 10
			small = base - 1
			(x1,y1) = ( int(smallW/base),int(smallH/base) )
			(x2,y2) = (int(smallW*small/base),int(smallH*small/base) )
			crop_center = crop_img[y1:y2,x1:x2]
			print(crop_center.shape[:2])
			h1 = gethist(crop_center)
			h1cnt = gethistcnt(crop_center)
			forethreshold = dynrange2
			backthreshold = dynrange4
			forecolor = topcolor(h1-h0,10)
			#backcolor = get_corner_color(crop_img,backthreshold)
			backcolor = (topcolor(cnt2pct(h0cnt-h1cnt),backthreshold))
			for c in backcolor:
				if c in forecolor:
					forecolor.remove(c)
			print("fore and back color")
			print(forecolor)
			print(backcolor)
			#cv2.rectangle(crop_img, (x1,y1), (x2,y2), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			#cv2.putText(crop_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


			(row,column,channel) = crop_img.shape
			edgestart = 100
			edgeend = 100
			edges = cv2.Canny(crop_img,edgestart,edgeend)
			
			objx = []
			objy = []
			
			for i in range(row):	
				for j in range(column):
					if(edges[i][j] > 0):
						objy.append(i)
						objx.append(j)
			#print(len(objx))
			#print(len(objy))	
			if (len(objx) == 0 or len(objy) == 0):
				continue
			a = np.array(objx)
			b = np.array(objy)
			mid=50
			midx = np.percentile(a, 50)
			midy = np.percentile(b, 50)
			dx = np.std(a)
			dy = np.std(b)
			dv = 0.1*dynrange3*math.sqrt(dx*dx+dy*dy)
			print("circle {} {} {}".format(int(midx), int(midy), int(dv) ))
			#cv2.circle(crop_img, (int(midx), int(midy)), int(dv), color, 2)
			
			print("mark background by color")
			marker = np.zeros_like(crop_img[:,:,0]).astype(np.int32)
			#  check if color bin = top colr marker = 255, else 1, 
			
			for i,j in itertools.product(range(row),range(column)):
				cim = [0,0,0]
				for k in range(3):
					cim[k] = index2bin(crop_img[i,j,k])
				for colortup in forecolor:
					(b,g,r) = colortup
					if( marker[i,j] == 0 and b == cim[0] and g == cim[1] and r == cim[2]):
						#bgravg = (sum([b,g,r])/3)*(256/3)+100
						#print(bgravg)
						#marker[i,j] = max(min(255,bgravg),100)
						marker[i,j] = 255
						#print(marker[i,j])
					bk = forethreshold
					(b,g,r) = crop_img[i,j]
					if( marker[i,j] == 0 and b <= bk and g <= bk and r <= bk ):
						marker[i,j] = 255
				#for colortup in backcolor:
					#(b,g,r) = colortup
					#if( marker[i,j] == 0 and b == cim[0] and g == cim[1] and r == cim[2] ):
						#marker[i,j] = 1
				for colortup in backcolor:
					(b,g,r) = colortup
					if(marker[i,j] == 0 and b == cim[0] and g == cim[1] and r == cim[2] ):
						marker[i,j] = 1
			

			final_img = mywater(crop_img,marker)
			final_img = bgtransparent(final_img)
			#final_img = cv2.resize(final_img, (int(smallW/ratio), int(smallH/ratio)) )
			
			#(smallH, smallW) = final_img.shape[:2]
			#ratio = max(smallW/640,smallH/480)
			#final_img = cv2.resize(final_img, (int(smallW/ratio), int(smallH/ratio)) )
			print("done hist cut")

			one_img['img'] = final_img
			one_img['cropim'] = crop_img
			final_list.append(one_img)
	return final_list
		
def graycutbg(image,dynrange1=25,dynrange2=25,dynrange3=25,dynrange4=25):
	# load the COCO class labels our YOLO model was trained on
	#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	print("[INFO] loading YOLO {} {} {} {}".format(dynrange1,dynrange2,dynrange3,dynrange4) )
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
		
	# derive the paths to the YOLO weights and model configuration
	#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	print("[INFO] loaded YOLO label")
	# derive the paths to the YOLO weights and model configuration
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	#image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	dynrange1 = max(1,dynrange1)
	ratio = max(max(W/1024,H/768)/dynrange1,1)
	print("image size = {} {} {} {}".format(H,W,dynrange1,ratio))
	#image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	#image = cv2.resize(image, (int(W/ratio), int(H/ratio)) )
	#h0 = gethist(image)
	#h0cnt = gethistcnt(image)
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)
		
	final_list=[]
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			print("processing {} {}".format(LABELS[classIDs[i]],i)) 
			one_img = {"key": i, "label": LABELS[classIDs[i]]}
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			# Rectange values: start x, start y, width, height
			rectangle = (x, y, w, h)
			print("size is {} {}".format(w,h))
			if ( w*h < 100 ) :
				continue
			crop_img = image[y:y+h, x:x+w]
			#crop_center = image[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4)]
			#print(crop_center.shape[:2])
			#h1 = gethist(crop_center)
			#h1cnt = gethistcnt(crop_center)
			#forethreshold = dynrange2
			#backthreshold = dynrange4
			#forecolor = topcolor(h1-h0,forethreshold)
			#backcolor = topcolor(cnt2pct(h0-h1),backthreshold)
			#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

			# show the output image
			#imS = cv2.resize(crop_img, (640, 480))
			(smallH, smallW) = crop_img.shape[:2]
			if ( smallW*smallH < 100 ) :
				continue
			print("image size = {} {}".format(smallH, smallW))
			ratio = max(smallW/640,smallH/480)
			img = crop_img
			grayim = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			#img = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )
			(row,column,channel) = img.shape
			edgestart = 100
			edgeend = 100
			edges = cv2.Canny(img,edgestart,edgeend)
			
			objx = []
			objy = []
			
			for i in range(row):	
				for j in range(column):
					if(edges[i][j] > 0):
						objy.append(i)
						objx.append(j)
			#print(len(objx))
			#print(len(objy))	
			if (len(objx) == 0 or len(objy) == 0):
				continue
			a = np.array(objx)
			b = np.array(objy)
			mid=50
			midx = np.percentile(a, 50)
			midy = np.percentile(b, 50)
			print("mid is {} {}".format(midx,midy))
			dx = np.std(a)
			dy = np.std(b)
			#dv = math.sqrt(dx*dx+dy*dy)
			dv = 0.1*dynrange3*min(dx,dy)
			#dv = 0.1*dynrange3*math.sqrt(dx*dx+dy*dy)
			print("dv = {}".format(dv))
			forecolor = []
			backcolor = []
			for i,j in itertools.product(range(row),range(column)):
				if( distance(j,i,midx,midy) <= dv):
					forecolor.append(grayim[i,j])
				elif(iscorner(i,j,row,column) ):
					backcolor.append(grayim[i,j])
			mincolor = float(dynrange4) / 100.0
			print(mincolor)
			hist1 = np.histogram(forecolor,bins=256,density=True)
			forecolor=np.argwhere(hist1[0] > 0.01)
			
			cornercolor = [grayim[0,0],grayim[0,column-1],grayim[row-1,0],grayim[row-1,column-1] ]
				
			print("mark background by color")
			marker = np.zeros_like(img[:,:,0]).astype(np.int32)
			#  check if color bin = top colr marker = 255, else 1, 
			for i,j in itertools.product(range(row),range(column)):
				if( marker[i,j] == 0 and grayim[i,j] in forecolor):
					marker[i,j] = 255
				if( marker[i,j] == 0 and grayim[i,j] <= 2):
					marker[i,j] = 255
				if( marker[i,j] == 0 and grayim[i,j] in backcolor):
					marker[i,j] = 1
				if( marker[i,j] == 0 and grayim[i,j] in cornercolor):
					marker[i,j] = 1

			print("mark background by gray")
 		
			
			
			marked = cv2.watershed(img, marker)
			# Make the background black, and what we want to keep white
			marked[marked == 1] = 0
			marked[marked > 1] = 255

			# Use a kernel to dilate the image, to not lose any detail on the outline
			# I used a kernel of 3x3 pixels
			kernel = np.ones((3,3),np.uint8)
			dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations = 1)

			# Plot again to check whether the dilation is according to our needs
			# If not, repeat by using a smaller/bigger kernel, or more/less iterations
			#plt.imshow(dilation, cmap='gray')
			#plt.show()

			# Now apply the mask we created on the initial image
			final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))
			
			
			
			(smallH, smallW) = final_img.shape[:2]
			ratio = max(smallW/640,smallH/480)
			final_img = cv2.resize(final_img, (int(smallW/ratio), int(smallH/ratio)) )
			crop_img = cv2.resize(crop_img, (int(smallW/ratio), int(smallH/ratio)) )
			print("done gray cut")
			tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
			_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
			b, g, r = cv2.split(final_img)
			rgba = [b,g,r, alpha]
			dst = cv2.merge(rgba,4)
			one_img['img'] = dst
			one_img['cropim'] = crop_img
			final_list.append(one_img)
	return final_list

@app.route('/image', methods=['POST'])
def getfile():
	if not request.json:
		abort(400)
	print(request.json)
	mydict = request.json
	for words in open("mydata.txt", 'r').readlines():
		print(words)
	print("start get image")
	client = storage.Client()
	# https://console.cloud.google.com/storage/browser/[bucket-id]/iwasnothing-ml-temp
	bucket = client.get_bucket('iwasnothing03.appspot.com')
	# Then do other things...
	blob = bucket.get_blob('public/'+mydict['file'])
	img_str=blob.download_as_string()
	print("blob get image")
	nparr = np.fromstring(img_str, np.uint8)
	print("array get image")
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	print("decode image")
	if (img_np is None):
		return jsonify(items = [])
		
	#imS = cv2.resize(img_np, (640, 480))
	if( mydict['method'] == 'grabcut'):
		print("grabcut")
		final_list = grabcutbg(img_np,int(mydict['parm1']),int(mydict['parm2']),int(mydict['parm3']),int(mydict['parm4']))
	elif( mydict['method'] == 'histcut'):
		print("histcut")
		final_list = histcutbg(img_np,int(mydict['parm1']),int(mydict['parm2']),int(mydict['parm3']),int(mydict['parm4']))
	else:
		final_list = removebg(img_np,int(mydict['parm1']),int(mydict['parm2']),int(mydict['parm3']),int(mydict['parm4']))
	list = []
	for im1 in final_list:
		print("showing {} {}".format(im1['key'],im1['label']) )
		imS = im1['img']
		ret, buf = cv2.imencode('.png',imS)
		print(ret) 
		#response = make_response(buf.tobytes())
		#response.headers['Content-Type'] = 'image/png'
		buf_str = np.array(buf).tostring()
		print("converted to string np array")
		fileprefix = mydict['file'].split('.')[0]
		outfilename = 'public/{}_{}_{}.png'.format(fileprefix,im1['key'],im1['label'])
		cropfilename = 'public/{}_{}_{}_crop.png'.format(fileprefix,im1['key'],im1['label'])
		list.append({'key': int(im1['key']), 'name': outfilename, 'crop': cropfilename})
		print("writing bucket " + outfilename)
		outblob = bucket.blob(outfilename)
		print("upload to bucket")
		outblob.upload_from_string(buf_str,content_type='image/png')
		outblob.make_public()
		print("done upload")
		# upload cropped img
		imC = im1['cropim']
		ret, bufC = cv2.imencode('.png',imC)
		print(ret) 
		#response = make_response(buf.tobytes())
		#response.headers['Content-Type'] = 'image/png'
		bufC_str = np.array(bufC).tostring()
		print("writing bucket " + cropfilename)
		outblobc = bucket.blob(cropfilename)
		print("upload to bucket")
		outblobc.upload_from_string(bufC_str,content_type='image/png')
		outblobc.make_public()
		print("done upload")
		
	print(list)   

	return jsonify(items = list)



if __name__ == "__main__":
    # app.run(host="localhost", debug=True)
    app.run(host="0.0.0.0", debug=True)

