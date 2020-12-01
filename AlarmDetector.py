import time
import TrainerProcess
import numpy as np
import Constants as const

class AlarmDetecter:

    #EAR
    BLINK_THRESHOLD = 25
    preCheckTime = None
    isFirstEarCheck = True

    earFrameCount = 0  # 눈 감았다고 판단되는 프레임 수, 계속 0으로 할당
    frameCount = 0  # 현재까지의 프레임 수, 0 ~ BLINK_CHECK_FRAME_NUMBER 까지
    blinkCount = 0  # 눈 감으면 값 오름

    HABIT_CHECK_DURATION = 5
    PROPER_BLINK_NUM_PER_SEC = 1 / 3

    #EYE TRAIN
    EYE_TRAIN_THRESHOLD = 15

    def init(self):
        self.preCheckTime = time.time()

    def shouldEarAlarmOn(self, trainer, facialLandmark, earThreshold, BLINK_CHECK_FRAME_NUMBER, BLINK_FRAME_THRESHOLD):
        if self.isFirstEarCheck:
            self.isFirstEarCheck = False
            self.preCheckTime = time.time()

        ear = trainer.getEyeAspectRatio(facialLandmark)

        if ear < earThreshold:
            self.earFrameCount += 1

        if self.frameCount >= BLINK_CHECK_FRAME_NUMBER:
            if self.earFrameCount >= BLINK_FRAME_THRESHOLD:
                self.blinkCount += 1
            self.earFrameCount = 0
            self.frameCount = 0

        self.frameCount += 1


        curTime = time.time()
        timeGapSec = curTime - self.preCheckTime
        res = None
        if timeGapSec >= self.HABIT_CHECK_DURATION:
            if self.blinkCount < int(self.PROPER_BLINK_NUM_PER_SEC * self.HABIT_CHECK_DURATION):
                res = [TrainerProcess.MessageType.EAR, True]
            else:
                res = [TrainerProcess.MessageType.EAR, False]

            self.preCheckTime = curTime
            self.blinkCount = 0

        return res

    def isOpenNextEyeTrain(self, eyeTrainPos, irisCenter, eyeCenter, eyeRect):

        eyeVector = np.array(irisCenter, dtype="float32") - np.array(eyeCenter, dtype="float32")
        eyeVector[1] *= -1
        eyeWidth = eyeRect[1][0] - eyeRect[0][0] - 2 * const.EYE_MARGIN['x']
        eyeHeight = eyeRect[1][1] - eyeRect[0][1] - 2 * const.EYE_MARGIN['y']

        accurate = 0

        pos = np.array(eyeTrainPos, dtype="float32")
        posUnit = pos / np.linalg.norm(pos)
        rawAccurate = np.dot(eyeVector, posUnit)

        if rawAccurate > 0:
            trainMaxVector = posUnit
            trainMaxVector[0] *= eyeWidth / 2
            trainMaxVector[1] *= eyeHeight / 2
            # print("eyeCenter", eyeCenter, "irisCenter", irisCenter)
            # print("posUnit", posUnit, "eyeVector", eyeVector)
            # print("raw accurate", rawAccurate)
            # print("pos Max Vector", trainMaxVector)
            # print(rawAccurate * rawAccurate, np.dot(trainMaxVector, trainMaxVector))
            accurate = rawAccurate * rawAccurate / np.dot(trainMaxVector, trainMaxVector) * 100 # to % unit
            print("[eye trainer] accurate", accurate)
            # print()
        if accurate > self.EYE_TRAIN_THRESHOLD:
            return [TrainerProcess.MessageType.EYE_TRAIN, True]
        else:
            return None




