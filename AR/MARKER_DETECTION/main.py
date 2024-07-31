import cv2 as cv
from cv2 import aruco
import numpy as np


marker_dic = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

cap  = cv.VideoCapture(0)


while True:
    success ,img = cap.read()

    if not success:
        break

    gray_img = cv.cvtColor(img ,cv.COLOR_BGR2GRAY)
    marker_corners ,marker_IDs ,reject = aruco.detectMarkers(
        gray_img ,marker_dic ,parameters=param_markers) 

    if marker_corners:

        for ids,corners in zip(marker_IDs ,marker_corners):
            
            cv.polylines(img ,[corners.astype(np.int32)], True ,(0,255,255) ,4 ,cv.LINE_AA)
            corners = corners.reshape(4,2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()

            cv.putText(img ,f"ID : {ids[0]}",
                        top_right ,cv.FONT_HERSHEY_COMPLEX ,1.3 ,(0,255,255),2,cv.LINE_AA)

            # print(corners.shape)
            # print(ids ," "  ,corners)

    cv.imshow("Frame" ,img)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()