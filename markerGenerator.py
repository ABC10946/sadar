import cv2
from cv2 import aruco
import os

markDir = './markers'
numMark = 50
sizeMark = 500

dictArco = aruco.Dictionary_get(aruco.DICT_4X4_50)

for count in range(numMark):
    idMark = count
    imgMark = aruco.drawMarker(dictArco, idMark, sizeMark)

    if count < 10:
        imgNameMark = 'mark_id_0' + str(count) + '.png'
    else:
        imgNameMark = 'mark_id_' + str(count) + '.png'
    
    pathMark = os.path.join(markDir, imgNameMark)
    print(pathMark)

    cv2.imwrite(pathMark, imgMark)
