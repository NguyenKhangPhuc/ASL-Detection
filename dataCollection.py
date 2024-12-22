import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
OFFSET = 30
IMAGE = 300
COUNT = 0
folder = "Data/A"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        print(x,y,w,h)
        imageWhite = np.ones((IMAGE, IMAGE, 3), np.uint8) * 255
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
        aspectRatio = h/w
        if aspectRatio > 1:
            w_calculated = math.ceil((IMAGE * w )/ h)
            imgResize = cv2.resize(imgCrop, (w_calculated, IMAGE))
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - w_calculated) / 2)
            imageWhite[0:imgResizeShape[0], center:imgResizeShape[1] + center] = imgResize
        else:
            h_calculated = math.ceil((IMAGE * h) / w)
            imgResize = cv2.resize(imgCrop, (IMAGE, h_calculated))
            imgResizeShape = imgResize.shape
            center = math.ceil((IMAGE - h_calculated) / 2)
            imageWhite[center:imgResizeShape[0] + center, 0:imgResizeShape[1]] = imgResize
        cv2.imshow("HandImage", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        COUNT+=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imageWhite)
        print(COUNT)
    if key == ord("q"):
        break