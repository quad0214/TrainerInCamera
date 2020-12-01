import Trainer
import cv2
import Constants
import time
import numpy as np

BLINK_THRESHOLD = 25
BLINK_FRAME_THRESHOLD = 1
BLINK_CHECK_FRAME_NUMBER = 7
preCheckTime = time.time()

earFrameCount = 0 # 눈 감았다고 판단되는 프레임 수, 계속 0으로 할당
frameCount = 0 # 현재까지의 프레임 수, 0 ~ BLINK_CHECK_FRAME_NUMBER 까지
blinkCount = 0 # 눈 감으면 값 오름

HABIT_CHECK_DURATION = 5
PROPER_BLINK_NUM_PER_SEC = 1 / 3

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainer = Trainer.Trainer()
    while True:
        bgr, img = trainer.getWebcamBGRAndGrayImage(480)
        img = cv2.flip(img, 1)

        #test
        bgr = cv2.flip(bgr, 1)

        rect = trainer.getFaceRect(img)
        if rect != None:
            facialLandmark = trainer.getFacialLandmark(img, rect)
            ear = trainer.getEyeAspectRatio(facialLandmark)

            ##### test
            leftEye = list(facialLandmark.part(i) for i in Constants.LEFT_EYE_INDEXES)
            leftEye = list((mark.x, mark.y) for mark in leftEye)

            for coor in leftEye:
                cv2.circle(bgr, coor, 2, (0, 0, 255), 1)
            #####

            if ear < BLINK_THRESHOLD:
                earFrameCount += 1

            if frameCount >= BLINK_CHECK_FRAME_NUMBER:
                if earFrameCount >= BLINK_FRAME_THRESHOLD:
                    blinkCount += 1
                earFrameCount = 0
                frameCount = 0

            frameCount += 1

            print(blinkCount)

            curTime = time.time()
            timeGapSec = curTime - preCheckTime
            if timeGapSec >= HABIT_CHECK_DURATION:
                if blinkCount < int(PROPER_BLINK_NUM_PER_SEC * HABIT_CHECK_DURATION):
                    #todo show alarm
                    print("문제 있음!!", blinkCount)

                preCheckTime = curTime
                blinkCount = 0


        cv2.imshow("image", bgr)
        if cv2.waitKey(1) > 0:
            break

    trainer.dispose()
    cv2.destroyAllWindows()

