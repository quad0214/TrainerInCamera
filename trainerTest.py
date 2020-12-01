import Trainer
import cv2
import Constants
import numpy as np
i = 0
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainer = Trainer.Trainer()
    while True:
        bgr, img = trainer.getWebcamBGRAndGrayImage(480)
        img = cv2.flip(img, 1)

        #test
        bgr = cv2.flip(bgr, 1)

        rect = trainer.getFaceRect(img)
        rotateMatrix = None
        tVec = None
        if rect != None:
            facialLandmark = trainer.getFacialLandmark(img, rect)
            # EAR
            #
            # trainer.getEyeAspectRatio(facialLandmark)
            # leftEye = list(facialLandmark.part(i) for i in Constants.LEFT_EYE_INDEXES)
            # leftEye = list((mark.x, mark.y) for mark in leftEye)
            # tempLeftEye = leftEye

            # eye, iris center
            #
            # minXY = []
            # maxXY = []
            # leftEye = np.array(leftEye)
            # eyeRect = [
            #     (
            #         min(leftEye[:, 0]) - Constants.EYE_MARGIN['x'],
            #         min(leftEye[:, 1]) - Constants.EYE_MARGIN['y']
            #     ),
            #     (
            #         max(leftEye[:, 0]) + Constants.EYE_MARGIN['x'],
            #         max(leftEye[:, 1]) + Constants.EYE_MARGIN['y']
            #     )]
            #
            # irisAndEyeCenterPos = trainer.getIrisAndEyeCenter(img, eyeRect)
            # if irisAndEyeCenterPos == None:
            #     continue
            #
            # irisCenterPos, eyeCenterPos = irisAndEyeCenterPos
            # print("irisCenterPos", irisCenterPos)
            # print("eyeCenterPos", eyeCenterPos)
            #
            # #for test
            # cv2.circle(bgr, irisCenterPos, 2, (0, 0, 255), 1)
            # cv2.circle(bgr, eyeCenterPos, 2, (0, 255, 255), 1)
            #
            #
            # for pos in tempLeftEye:
            #     cv2.circle(bgr, pos, 1, (0, 255, 0), 1)

            # get World coor
            # if rotateMatrix == None and tVec == None:
            #     rotateMatrix, tVec = trainer.getRotateMatrixAndTransVec(facialLandmark)
            #
            # coors = trainer.getObjectsWorldCoors(facialLandmark, [8, 36], rotateMatrix, tVec)
            # print(f"{i} {coors[0][2] - coors[1][2]}")
            # for index in [8, 36]:
            #     part = facialLandmark.part(index)
            #     coor = (part.x, part.y)
            #     cv2.circle(bgr, coor, 1, (0, 0, 255), 1)
            #
            # i += 1


        cv2.imshow("image", bgr)
        if cv2.waitKey(1) > 0:
            break

    trainer.dispose()
    cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
