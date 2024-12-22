import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import  Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
OFFSET = 30
IMAGE = 300
COUNT = 0
labels = ["A", "B", "C", "D", "E", "F",
          "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R",
          "S", "T", "U", "V", "W", "X",
          "Y", "Z"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imageWhite = np.ones((IMAGE, IMAGE, 3), np.uint8) * 255
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
        aspectRatio = h/w
        if aspectRatio > 1:
            w_calculated = math.ceil((IMAGE * w )/ h)
            imgResize = cv2.resize(imgCrop, (w_calculated, IMAGE))
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - w_calculated) / 2)
            imageWhite[0:imgResizeShape[0], center:imgResizeShape[1] + center] = imgResize
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        else:
            h_calculated = math.ceil((IMAGE * h) / w)
            imgResize = cv2.resize(imgCrop, (IMAGE, h_calculated))
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - h_calculated) / 2)
            imageWhite[center:imgResizeShape[0] + center, 0:imgResizeShape[1]] = imgResize
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        cv2.imshow("HandImage", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)
        cv2.putText(imgOutput, labels[index], (x + 50, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
