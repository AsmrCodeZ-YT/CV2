import cv2 as cv
from cv2 import aruco 
import numpy as np
import os 
import ArucoModule as arm


cap = cv.VideoCapture(0)
augDicts = arm.loadAugImage("Image")
while True:
    success ,img  = cap.read()
    arucoFound = arm.findArucoMarkers(img)

    # Loop through all markers and AR 
    if len(arucoFound[0]) != 0:
        for bbox ,id  in zip(arucoFound[0] , arucoFound[1]):
            if int(id) in augDicts.keys():
                img = arm.argumentAruco(bbox , id ,img ,augDicts[int(id)])
                
    cv.imshow("image" ,img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break




