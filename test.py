import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Hand Detector for up to 2 hands
detector = HandDetector(maxHands=2)

# Load the trained model and labels
classifier = Classifier("Model/keras_model.keras", "Model/labels.txt")

# Parameters
offset = 20
imgSize = 224

# Automatically read the labels from the labels.txt file
with open("Model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Create a resizable OpenCV window
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Flip the image horizontally to create a mirror effect
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    left_hand_output = None
    right_hand_output = None
    combined_output = None

    if hands:
        hand_outputs = {}
        for hand in hands:
            # Swap the roles of left and right hands
            hand_type = 'Right' if hand['type'] == 'Left' else 'Left'

            x, y, w, h = hand['bbox']

            # Crop and process the hand
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Get prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                hand_outputs[hand_type] = (labels[index], prediction[index])

                # Overlay label
                text_y = max(30, y - offset - 10)  # Adjust text position above the bounding box
                cv2.putText(imgOutput, f"{hand_type}: {labels[index]} ({prediction[index]:.2f})",
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Draw bounding box
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 0), 2)

        # Separate outputs for left and right hands
        left_hand_output = hand_outputs.get('Left', None)
        right_hand_output = hand_outputs.get('Right', None)

        # Combined output if both hands are present
        if 'Left' in hand_outputs and 'Right' in hand_outputs:
            # Merge bounding boxes
            left_hand = hands[0] if hands[0]['type'] == 'Left' else hands[1]
            right_hand = hands[1] if hands[0]['type'] == 'Left' else hands[0]

            # Crop combined region
            x_min = max(0, min(left_hand['bbox'][0], right_hand['bbox'][0]) - offset)
            y_min = max(0, min(left_hand['bbox'][1], right_hand['bbox'][1]) - offset)
            x_max = min(img.shape[1], max(left_hand['bbox'][0] + left_hand['bbox'][2],
                                          right_hand['bbox'][0] + right_hand['bbox'][2]) + offset)
            y_max = min(img.shape[0], max(left_hand['bbox'][1] + left_hand['bbox'][3],
                                          right_hand['bbox'][1] + right_hand['bbox'][3]) + offset)

            imgCrop = img[y_min:y_max, x_min:x_max]
            if imgCrop.size > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = (y_max - y_min) / (x_max - x_min)

                if aspectRatio > 1:
                    k = imgSize / (y_max - y_min)
                    wCal = math.ceil(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / (x_max - x_min)
                    hCal = math.ceil(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Get combined prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                combined_output = (labels[index], prediction[index])

                # Overlay combined label in the top-left corner
                cv2.putText(imgOutput, f"Combined: {labels[index]} ({prediction[index]:.2f})",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw combined bounding box
                cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the camera feed
    cv2.imshow('Image', imgOutput)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
