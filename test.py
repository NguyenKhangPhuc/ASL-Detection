"""
This module captures hand gestures from a webcam, detects a single hand, and classifies the gesture using a pre-trained model.

Modules and Libraries:
- cv2: OpenCV for video capture and image processing.
- numpy: Used for creating a blank white image (canvas).
- math: Provides mathematical functions like ceil for resizing calculations.
- time: Used for generating unique filenames.
- cvzone.HandTrackingModule: For detecting and tracking hands in the video feed.

Features:
1. Detects a hand using HandDetector.
2. Crops and resizes the hand region to fit a 300x300 white canvas.
3. Classifies the gesture with a pre-trained model (`keras_model.h5`) and displays the result.
4. Exits when the 'q' key is pressed.

Constants:
- OFFSET: Padding around the hand bounding box.
- IMAGE: Canvas size (300x300 pixels).
- labels: List of hand sign labels corresponding to model predictions.

The model struggles to accurately recognize hand gestures corresponding to the letters:
- E,S,M,N, and T
- V and K and W.

Suitable Version:
Python 3.10.8
cvzone 1.6.1
mediapipe 0.10.5
tensorflow 2.9.1
"""

import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import  Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#Get the classifier from a pretrained_model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
OFFSET = 30
IMAGE = 300
labels = ["A", "B", "C", "D", "E", "F",
          "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R",
          "S", "T", "U", "V", "W", "X",
          "Y", "Z"]
while True:
    # Find the camera
    success, img = cap.read()
    # Copy the current image scene to display it in the future.
    imgOutput = img.copy()
    # Find the hand in the camera
    hands, img = detector.findHands(img)
    if hands:
        # If there are any hand in the camera
        # Get its height, width, and x y position
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        # Filter out the hand and put it in another the window
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
        # Put the filtered hand into a 300x300 white window
        imageWhite = np.ones((IMAGE, IMAGE, 3), np.uint8) * 255
        aspectRatio = h/w
        if aspectRatio > 1:
            # If height > width => height = 300, calculate the width.
            # Resize the imgCrop to the new size.
            w_calculated = math.ceil((IMAGE * w )/ h)
            imgResize = cv2.resize(imgCrop, (w_calculated, IMAGE))
            # Get the resize image shape and put it in the center of the 300x300 window
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - w_calculated) / 2)
            imageWhite[0:imgResizeShape[0], center:imgResizeShape[1] + center] = imgResize
            # Predict the current gesture displayed in the 300x300 window.
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        else:
            # If width > height => width = 300, calculate the height.
            # Resize the imgCrop to the new size.
            h_calculated = math.ceil((IMAGE * h) / w)
            imgResize = cv2.resize(imgCrop, (IMAGE, h_calculated))
            # Get the resize image shape and put it in the center of the 300x300 window
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - h_calculated) / 2)
            imageWhite[center:imgResizeShape[0] + center, 0:imgResizeShape[1]] = imgResize
            # Predict the current gesture displayed in the 300x300 window.
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        # Display the filtered hand and 300x300 window
        cv2.imshow("HandImage", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)
        # Display the predicted alphabet base on the last prediction.
        cv2.putText(imgOutput, labels[index], (x + 50, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    # Display the full camera window
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):
        #if key "q" is pressed, stop the program.
        break
