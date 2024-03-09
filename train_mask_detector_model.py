# import the necessary packages
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

Directory_path = r"C:\Users\anike\OneDrive\Documents\HIS\Sem-2\RT\RealTime-Face-Mask-Detection\dataset"
Categories_path = ["with_mask", "without_mask"]

data = []
labels = []
#Load images and start training
print("Loading images to train on...")

#loop through all images from both categories
for category in Categories_path:
	#Create a path of an image
    path = os.path.join(Directory_path, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

# Do one-hot encoding on the labels to create categorical variable
lbl = LabelBinarizer()
labels = lbl.fit_transform(labels)
labels = to_categorical(labels)

# initialize the initial learning rate for model, number of epochs and batch size
Initial_learning_rate = 1e-4
Epochs = 20
batch_size = 32

data = np.array(data, dtype="float32")
labels = np.array(labels)

#split data into train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
img_augmentation = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network
base_Model = MobileNetV2(weights="imagenet", include_top=False,
						 input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed at the start of the model
head_Model = base_Model.output
head_Model = AveragePooling2D(pool_size=(7, 7))(head_Model)
head_Model = Flatten(name="flatten")(head_Model)
head_Model = Dense(128, activation="relu")(head_Model)
head_Model = Dropout(0.5)(head_Model)
head_Model = Dense(2, activation="softmax")(head_Model)

# place the head model on top of the base model
model = Model(inputs=base_Model.input, outputs=head_Model)

# Freeze Base model
for layer in base_Model.layers:
	layer.trainable = False

# compile  model
print("compiling model...")
opt = Adam(learning_rate=Initial_learning_rate)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the model
print("Training head of the network...")
H = model.fit(
	img_augmentation.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=Epochs)

# make predictions on the testing set
print("Evaluating network accuracy...")
predIds = model.predict(testX, batch_size=batch_size)
predIds = np.argmax(predIds, axis=1)

# Print classification report
print(classification_report(testY.argmax(axis=1), predIds,
							target_names=lbl.classes_))

print("Saving model on machine...")
model.save("face_mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training - Loss and Accuracy Matrix ")
plt.xlabel("Epoch Number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("AccuracyMatrix.png")