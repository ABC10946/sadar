import cv2
from cv2 import aruco
import numpy as np

# webcam parameters
# cameraMatrix = np.array([[962.27928045, 0., 390.31456086],
#                          [0., 962.87080911, 284.98233988],
#                          [0., 0., 1.]])

# distCoeffs = np.array([-1.85295083e-01, 2.81974206e+00, 1.15101375e-03, 2.98202533e-03, -1.65586083e+01])

# pixel 5a back camera parameters
cameraMatrix = np.array(
    [[508.08563138,   0.  ,       317.93414728],
 [  0.        , 510.02650753, 240.67870614],
 [  0.        ,   0.      ,     1.        ]]
)


distCoeffs = np.array([ 0.16271226, -0.95617855, -0.00379688,  0.00348429,  2.12451617])


def main():
    cap = cv2.VideoCapture(2)

    dictAruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()


    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictAruco, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                print(ids[i])
                rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corner, 0.05, cameraMatrix, distCoeffs)
                tvec = tvec.flatten()
                rvec = rvec.flatten()
                rvecMatrix = cv2.Rodrigues(rvec)
                rvecMatrix = rvecMatrix[0]

                transposeTvec = tvec[np.newaxis, :].T
                projectionMatrix = np.hstack((rvecMatrix, transposeTvec))
                eulerAngle = cv2.decomposeProjectionMatrix(projectionMatrix)[6]

                if ids[i].flatten() == 0:
                    print("x:", tvec[0])
                    print("y:", tvec[1])
                    print("z:", tvec[2])
                    print("roll:", eulerAngle[0])
                    print("pitch:", eulerAngle[1])
                    print("yaw:", eulerAngle[2])
                
                aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
