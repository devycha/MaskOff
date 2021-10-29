# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
from smbus2 import SMBus
from mlx90614 import MLX90614
from sound import sound_sync
from sound import sound_no_sync
import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import os
import pyzbar.pyzbar as pyzbar
import motor

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)

try: 
  f = open('qrcode_data.txt', 'r', encoding='utf8')
  data_list = f.readlines()
except FileNotFoundError:
  pass

# vs = VideoStream(src=0).start()
# time.sleep(2.0)
cap = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# frame = vs.read()
	# frame = imutils.resize(frame, width=400)
	success, frame = cap.read()

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	
	mask_on = False
	check = False
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# label = "Mask" if mask > 0.995 else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		if mask > 0.985:
			temp = round(sensor.get_object_1(), 2)
			print("현재 온도는 ", temp, "'C 입니다.")
			if temp > 37.4 :
				label = "High Temperature"
				color = (0, 0, 255)
			elif temp < 30:
				# sound_no_sync.sound("./sound/tempcheck.wav")
				label = "Temperature Check"
				color = (0, 0, 255)
			else:
				label = "Mask On"
				color = (0, 255, 0)
				mask_on = True
		else :
			label = "Mask Off"
			color = (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	if mask_on:
		print("[INFO] qr check is ready")
		qr = True
		while (qr):
			success, frame = cap.read()
			if success:
				cv2.putText(frame, "QR Check", (300, 125 - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
				cv2.rectangle(frame, (225, 125), (425, 325), (0, 255, 0), 2)
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				for code in pyzbar.decode(frame):
					my_code = code.data.decode('utf-8')
					sound_sync.sound("./sound/qr_sound2.wav")
					print('인식 성공', my_code)
					for i in range(len(data_list)):
						if my_code == data_list[i].rsplit('\n')[0]:
							check = True
							mask_on = False
							qr = False
						if i == len(data_list)-1:
							qr = False
							mask_on = False
		if check:
			sound_sync.sound("./sound/welcome.wav")
			motor.move()
		#TODO: 모터 개폐 시간 조절
   
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("t"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
