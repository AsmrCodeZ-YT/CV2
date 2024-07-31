import cv2 as cv
from cv2 import aruco


marker_dic = aruco.Dictionary_get(aruco.DICT_4X4_50)

MARKER_SIZE = 400 # pixels

for i in range(20):
    marker_img = aruco.drawMarker(marker_dic ,i ,MARKER_SIZE)
    cv.imshow("img" ,marker_img)
    cv.imwrite(f"GENERATE_MARKERS\markers_img\marker_{i}.png" ,marker_img)
    # cv.waitKey(0)
    # break

















