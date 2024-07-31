import cv2 as cv
from cv2 import aruco 
import numpy as np
import os 

def loadAugImage(path):
    myList = os.listdir(path)
    # print(myList)
    # print(len(myList))
    augDicts = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv.imread(f"{path}/{imgPath}")
        augDicts[key] = imgAug

    return augDicts


def findArucoMarkers(img ,markerSize=4 ,totalMarkers=50  ,draw=True):

    imgGray = cv.cvtColor(img ,cv.COLOR_BGR2GRAY)
    # arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    key = getattr(aruco ,f"DICT_{markerSize}X{markerSize}_{totalMarkers}")
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bbox ,ids ,rejected = aruco.detectMarkers(imgGray ,arucoDict ,parameters=arucoParam) 
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img ,bbox)

    return [bbox ,ids]

def argumentAruco(bbox ,id ,img ,imgAug , drawId=True):
    
    tl = bbox[0][0][0] ,bbox[0][0][1]
    tr = bbox[0][1][0] ,bbox[0][1][1]
    br = bbox[0][2][0] ,bbox[0][2][1]    
    bl = bbox[0][3][0] ,bbox[0][3][1]

    h ,w ,c = imgAug.shape

    pts1 = np.array([tl ,tr ,br ,bl])
    pts2 = np.float32([[0 ,0],[w ,0],
                        [w ,h],[0 ,h]])

    matrix ,_ = cv.findHomography(pts2 , pts1)
    imgOut = cv.warpPerspective(imgAug ,matrix ,(img.shape[1] ,img.shape[0])) 
    cv.fillConvexPoly(img , pts1.astype(int) ,(0 ,0 ,0))
    imgOut = img + imgOut

    # if drawId:
        # cv.putText(imgOut ,str(id) ,(tl[0],tl[1]) , cv.FONT_HERSHEY_PLAIN ,2 ,(255,0,255) ,2)

    return imgOut


def main():
    cap = cv.VideoCapture(0)
    augDics = loadAugImage("USEIT/Image")

    while True:
        success ,img  = cap.read()

        # arugi = cv.imread("USEIT\Image\9.jpg")
        arucoFound = findArucoMarkers(img)

        # # Loop through all markers and AR 
        if len(arucoFound[0]) != 0:
            for bbox ,id  in zip(arucoFound[0] , arucoFound[1]):
                if int(id) in augDics.keys():
                    img = argumentAruco(bbox , id ,img ,augDics[int(id)])
                
        cv.imshow("image" ,img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

            
    
if __name__ == "__main__":
    main()
    