import cv2
import numpy as np
import time

"""
        legacy_MultiTracker
        legacy_Tracker
        legacy_TrackerBoosting
        legacy_TrackerCSRT
        legacy_TrackerKCF
        legacy_TrackerMIL
        legacy_TrackerMOSSE
        legacy_TrackerMedianFlow
        legacy_TrackerTLD
"""
# you might need to delete legacy part like cv2.TrackerCSRT_create
trackers = {
    'csrt' : cv2.legacy.TrackerCSRT_create,  # hight accuracy ,slow
    'mosse' : cv2.legacy.TrackerMOSSE_create,  # fast, low accuracy
    'kcf' : cv2.legacy.TrackerKCF_create,   # moderate accuracy and speed
    'medianflow' : cv2.legacy.TrackerMedianFlow_create, # good but slow
    'mil' : cv2.legacy.TrackerMIL_create, # kinda bad
    'tld' : cv2.legacy.TrackerTLD_create, # very bad
    'boosting' : cv2.legacy.TrackerBoosting_create, # very bad
    'multi' : cv2.legacy.MultiTracker_create
}


cap = cv2.VideoCapture("vid0.mp4")
ret, frame = cap.read()

bbint=(163, 112, 537, 396)
# Only consider the region within the bounding box `bb`
bb_roi = frame[bbint[1]:bbint[1]+bbint[3], bbint[0]:bbint[0]+bbint[2]]
hsv = cv2.cvtColor(bb_roi, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV color space
lower_blue = np.array([ 90,  12, 204])
upper_blue = np.array([110, 112, 304])

# Threshold the image to create a binary mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw a bounding rectangle around the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)
x += bbint[0]
y += bbint[1]

cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


tracker_key = 'csrt'


bb= x, y, w, h
tracker = trackers[tracker_key]()
tracker=cv2.legacy.Tracker()


tracker.init(frame,bb)

while True:
    time.sleep(0.1)
    ret, frame = cap.read()

    if not ret:
        break

    success, bb = tracker.update(frame)

    if success:
        p1 = (int(bb[0]), int(bb[1]))
        p2 = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(5)

#tracker=cv2.legacy.Tracker
