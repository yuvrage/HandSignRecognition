import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Verify the camera is accessible
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Initialize the hand detector with a maximum of 2 hands detected
detector = HandDetector(maxHands=2)

# Parameters
offset = 20
imgSize = 300
counter = 0

# Specify the folder to save images
folder = "data/ThankYou"

# Create the folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Get existing image numbers if any
existing_files = os.listdir(folder)
existing_image_numbers = [
    int(f.split('_')[1].split('.')[0]) for f in existing_files if f.endswith('.jpg')
]

if existing_image_numbers:
    counter = max(existing_image_numbers) + 1
else:
    counter = 0  # Start from 0 if no images exist yet

print(f"Starting with image {counter}")

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        # Initialize a blank white image for the final output
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calculate the combined bounding box of both hands
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0

        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        # Ensure coordinates are within the image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)

        # Crop the region containing both hands
        imgCrop = img[y_min:y_max, x_min:x_max]

        if imgCrop.size > 0:  # Ensure the crop is valid
            aspectRatio = (y_max - y_min) / (x_max - x_min)

            try:
                if aspectRatio > 1:
                    # Height is greater than width
                    k = imgSize / (y_max - y_min)
                    wCal = math.ceil(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    # Width is greater than height
                    k = imgSize / (x_max - x_min)
                    hCal = math.ceil(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Display the cropped and resized images
                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

            except Exception as e:
                print(f"Error during resizing: {e}")

    # Display the original frame
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        # Save the image
        try:
            image_filename = f'{folder}/Image_{counter}.jpg'
            cv2.imwrite(image_filename, imgWhite)
            print(f"Image {counter} saved.")
            counter += 1  # Increment the counter for the next image
        except Exception as e:
            print(f"Error saving image: {e}")

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        print("Exiting...")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
