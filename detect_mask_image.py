from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


class MaskImg:
    def __init__(self, image, face='face_detector', model='mask_detector.model', confidence=0.5):
        self.tools = {
            'images': image,
            'face': face,
            'model': model,
            'confidence': confidence
        }

    def run(self):
        prototxt = os.path.sep.join([self.tools["face"], "deploy.prototxt"])
        weights = os.path.sep.join([self.tools["face"],
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxt, weights)

        model = load_model(self.tools["model"])


        image = cv2.imread(self.tools["images"])
        orig = image.copy()
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # loop over the detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.tools["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (mask, withoutMask) = model.predict(face)[0]

                label = "Mask" if mask > withoutMask else "Intruder"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Output", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    maskImg = MaskImg('images/test.JPG')
    maskImg.run()
