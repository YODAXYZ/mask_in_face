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
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
	
	
class MaskTrain:
	def __init__(self, dataset, plot='plot.png', model='mask_detector.model'):
		self.tools = {
			'dataset': dataset,
			'model': model,
			'plot': plot
		}
		
	def run(self):
		imagePaths = list(paths.list_images(self.tools["dataset"]))
		data = []
		labels = []

		for imagePath in imagePaths:
			label = imagePath.split(os.path.sep)[-2]

			# load the input images (224x224) and preprocess it
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)
			image = preprocess_input(image)

			data.append(image)
			labels.append(label)

		# convert the data and labels to NumPy arrays
		data = np.array(data, dtype="float32")
		labels = np.array(labels)

		lb = LabelBinarizer()
		labels = lb.fit_transform(labels)
		labels = to_categorical(labels)

		# partition the data into training and testing splits using 80% of
		# the data for training and the remaining 20% for testing
		(trainX, testX, trainY, testY) = train_test_split(data, labels,
														  test_size=0.20, stratify=labels, random_state=42)

		# construct the training images generator for data augmentation
		aug = ImageDataGenerator(
			rotation_range=20,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
			fill_mode="nearest")

		baseModel = MobileNetV2(weights="imagenet", include_top=False,
								input_tensor=Input(shape=(224, 224, 3)))

		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(128, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)

		# place the head FC model on top of the base model
		model = Model(inputs=baseModel.input, outputs=headModel)

		for layer in baseModel.layers:
			layer.trainable = False

		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
					  metrics=["accuracy"])

		# train the head of the network
		H = model.fit(
			aug.flow(trainX, trainY, batch_size=BS),
			steps_per_epoch=len(trainX) // BS,
			validation_data=(testX, testY),
			validation_steps=len(testX) // BS,
			epochs=EPOCHS)

		predIdxs = model.predict(testX, batch_size=BS)

		# find largest predicted probability
		predIdxs = np.argmax(predIdxs, axis=1)

		# classification report
		print(classification_report(testY.argmax(axis=1), predIdxs,
									target_names=lb.classes_))
		self.save_plot(model, H)

	def save_plot(self, model, head):
		model.save(self.tools["model"], save_format="h5")

		N = EPOCHS
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), head.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), head.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), head.history["acc"], label="train_acc")
		plt.plot(np.arange(0, N), head.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(self.tools["plot"])


if __name__ == '__main__':
	mask_train = MaskTrain('dataset/')
	mask_train.run()
