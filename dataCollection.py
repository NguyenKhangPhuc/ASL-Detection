"""
This module captures hand gestures using a webcam and processes them for hand sign data collection.
It detects a hand, crops the hand region from the video frame, resizes it to fit within a fixed
canvas size, and saves the processed image upon user input. The module uses the cvzone.HandTrackingModule
for hand detection and OpenCV for image processing.

Modules and Libraries:
- cv2: OpenCV for video capture and image processing.
- numpy: Used for creating a blank white image (canvas).
- math: Provides mathematical functions like ceil for resizing calculations.
- time: Used for generating unique filenames.
- cvzone.HandTrackingModule: For detecting and tracking hands in the video feed.

Workflow:
1. Captures video from the webcam.
2. Detects a single hand in the video frame using HandDetector.
3. Extracts the bounding box of the detected hand.
4. Crops the hand region with an offset to include some background.
5. Resizes the cropped image to fit within a 300x300 pixel white canvas while maintaining aspect ratio.
6. Displays the original, cropped, and resized images.
7. Saves the processed image to the specified folder when the 's' key is pressed.
8. Exits the application when the 'q' key is pressed.

Constants:
- OFFSET: Padding around the hand bounding box for cropping.
- IMAGE: Size of the white canvas (300x300 pixels).
- COUNT: Counter for the number of saved images.
- folder: Path to the folder where images will be saved.

Key Bindings:
- 's': Saves the processed hand image to the folder.
- 'q': Quits the application.
"""
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
# You can change the path right here to /B or /C,...
# then run and save your suitable image to the respective hand sign folder
folder = "Data/A"
# You then can go to the "Teachable Machine" website to get the exported model there
# Put it into the Model folder.
while True:
    #Find the camera
    success, img = cap.read()
    #Find the hand in the camera
    hands, img = detector.findHands(img)
    if hands:
        #If there are any hand in the camera
        #Get its height, width, and x y position
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        #Filter out the hand and put it in another the window
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
        #Put the filtered hand into a 300x300 white window
        imageWhite = np.ones((IMAGE, IMAGE, 3), np.uint8) * 255
        aspectRatio = h/w
        if aspectRatio > 1:
            #If height > width => height = 300, calculate the width.
            #Resize the imgCrop to the new size.
            w_calculated = math.ceil((IMAGE * w )/ h)
            imgResize = cv2.resize(imgCrop, (w_calculated, IMAGE))
            imgResizeShape = imgResize.shape
            #Get the resize image shape and put it in the center of the 300x300 window
            center = math.ceil((IMAGE - w_calculated) / 2)
            imageWhite[0:imgResizeShape[0], center:imgResizeShape[1] + center] = imgResize
        else:
            # If width > height => width = 300, calculate the height.
            # Resize the imgCrop to the new size.
            h_calculated = math.ceil((IMAGE * h) / w)
            imgResize = cv2.resize(imgCrop, (IMAGE, h_calculated))
            imgResizeShape = imgResize.shape
            # Get the resize image shape and put it in the center of the 300x300 window
            center = math.ceil((IMAGE - h_calculated) / 2)
            imageWhite[center:imgResizeShape[0] + center, 0:imgResizeShape[1]] = imgResize
        #Display the filtered hand and 300x300 window
        cv2.imshow("HandImage", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)
    #Display the full camera window
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        #if key "s" is pressed, get the 300x300 window image and
        #save it to the respective path
        COUNT+=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imageWhite)
        print(COUNT)
    if key == ord("q"):
        #if key "q" is pressed, stop the program.
        break