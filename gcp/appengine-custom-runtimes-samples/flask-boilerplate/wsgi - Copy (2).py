from flask import Flask
from flask import make_response
import numpy as np
import time
import cv2
from datetime import datetime
from google.cloud import storage
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app)



@app.route("/")
def hello():
    return "Docker Hello World!"

@app.route('/now')
def shownow():
	return str(datetime.now())

def removebg(image,dynval=25):
	print("[INFO] loading YOLO ")
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
	print("[INFO] loaded YOLO label")
	# derive the paths to the YOLO weights and model configuration
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	print("[INFO] done load YOLO from bucket...")
	# load our input image and grab its spatial dimensions
	#image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]

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
			crop_img = image[y:y+h, x:x+w]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			print(LABELS[classIDs[i]])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
				
							
							
			# show the output image
			imS = cv2.resize(crop_img, (640, 480))
			print("canny edge detection")
			img = imS

			edges = cv2.Canny(img,100,200)

			(row,column,channel) = img.shape
			print("remove background")
			marker = np.zeros_like(img[:,:,0]).astype(np.int32)


			objx = []
			objy = []
			for i in range(row):	
				for j in range(column):
					if(edges[i][j] > 0):
						objx.append(i)
						objy.append(j)
						
			a = np.array(objx)
			b = np.array(objy)
			mid=50
			fixrange = 20
			dynrange = 25
			pct1 = mid - dynrange
			pct2 = mid + dynrange
			pct3 = max(5,pct1 - fixrange)
			pct4 = min(95,pct2 + fixrange)
			px1 = np.percentile(a, pct1)
			py1 = np.percentile(b, pct1)
			px2 = np.percentile(a, pct2)
			py2 = np.percentile(b, pct2)
			px3 = np.percentile(a, pct3)
			py3 = np.percentile(b, pct3)
			px4 = np.percentile(a, pct4)
			py4 = np.percentile(b, pct4)


			print("obj is {},{},{},{}".format(px1,px2,py1,py2) )

			for i in range(row):	
				for j in range(column):
				# please add condition to reject too far away points
					if (i > px1 and i < px2 and j > py1 and j < py2):
						marker[i][j] = 255
					elif (i < px3 or i > px4 or j < py3 or j > py4):
						marker[i][j] = 1
					else:
						marker[i][j] = 0
			marked = cv2.watershed(img, marker)

			# Plot this one. If it does what we want, proceed;
			# otherwise edit your markers and repeat
			#plt.imshow(marked, cmap='gray')
			#plt.show()

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
			print("mask")
			# Now apply the mask we created on the initial image
			final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))
			print("transpartent")
			# cv2.imread reads the image as BGR, but matplotlib uses RGB
			# BGR to RGB so we can plot the image with accurate colors
			#b, g, r = cv2.split(final_img)
			#final_img = cv2.merge([r, g, b])
			#image = cv2.imread('./Desktop/maze.png')
			#final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGBA)
			#final_img[np.all(final_img == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
			tmp = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
			_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
			b, g, r = cv2.split(final_img)
			rgba = [b,g,r, alpha]
			dst = cv2.merge(rgba,4)
			one_img['img'] = dst
			final_list.append(one_img)
				
	return final_list
	
@app.route('/image')
def getfile():
	for words in open("mydata.txt", 'r').readlines():
		print(words)
	print("start get image")
	client = storage.Client()
	# https://console.cloud.google.com/storage/browser/[bucket-id]/iwasnothing-ml-temp
	bucket = client.get_bucket('iwasnothing03.appspot.com')
	# Then do other things...
	blob = bucket.get_blob('public/momo02.jpg')
	img_str=blob.download_as_string()
	print("blob get image")
	nparr = np.fromstring(img_str, np.uint8)
	print("array get image")
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	print("decode image")
	#imS = cv2.resize(img_np, (640, 480))
	final_list = removebg(img_np,25)
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
		outfilename = 'public/momo_crop_{}_{}.png'.format(im1['key'],im1['label'])
		list.append({'key': int(im1['key']), 'name': outfilename})
		print("writing bucket " + outfilename)
		outblob = bucket.blob(outfilename)
		print("upload to bucket")
		outblob.upload_from_string(buf_str,content_type='image/png')
		print("done upload")
	print(list)   

	return jsonify(items = list)



if __name__ == "__main__":
    # app.run(host="localhost", debug=True)
    app.run(host="0.0.0.0", debug=True)

