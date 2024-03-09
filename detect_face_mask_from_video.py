# import the necessary packages
import numpy as np
import imutils
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

#This function is used to detect and return the faces with their location
def detect_and_predict_face_mask(frame, faceNet, maskNet):
	# get the dimensions of the frame and then construct a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces
	faces = []
	locs = []
	preds = []

	# loop over the detected faces
	for j in range(0, detections.shape[2]):

		confidenceLevel = detections[0, 0, j, 2]
		if confidenceLevel > 0.5:

			box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

# load model for face detection
face_prototxtPath = r"C:\Users\anike\OneDrive\Documents\HIS\Sem-2\RT\RealTime-Face-Mask-Detection\face_detector\deploy.prototxt"
face_weightsPath = r"C:\Users\anike\OneDrive\Documents\HIS\Sem-2\RT\RealTime-Face-Mask-Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(face_prototxtPath, face_weightsPath)

# load the face mask detection model
facemaskNet = load_model(r"C:\Users\anike\OneDrive\Documents\HIS\Sem-2\RT\RealTime-Face-Mask-Detection\face_mask_detector.model")

# initialize the video stream with source as laptop camera
print("Starting video stream...")
videoStrm = VideoStream(src=0).start()

while True:
	# resize frame
	frame = videoStrm.read()
	frame = imutils.resize(frame, width=800)
	print(frame)
	# find faces in the frame
	(locs, preds) = detect_and_predict_face_mask(frame, faceNet, facemaskNet)

	for (boundin_box, pred) in zip(locs, preds):
		# get bounding box and predictions
		(startX, startY, endX, endY) = boundin_box
		(mask, withoutMask) = pred

		# determine the class label and color used for bounding box
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label shown
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# display the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `e` key was pressed, stop processing
	if key == ord("e"):
		break
cv2.destroyAllWindows()
videoStrm.stop()