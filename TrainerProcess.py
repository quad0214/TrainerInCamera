import cv2
import Constants as const
import time
from enum import Enum
import Trainer
import AlarmDetector
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import keyboard
import mouse


# multiprocessing을 위해서 모두 전역으로 사용

trainer = Trainer.Trainer()

# EAR
BLINK_FRAME_THRESHOLD = 1
BLINK_CHECK_FRAME_NUMBER = 10
PROPER_BLINK_NUM_PER_SEC = 1 / 3


# EYE_TRAIN

# DISTANCE
DISTANCE_CHECK_FRAME_NUMBER = 10
curDistanceCheckFrame = 0
distanceSum = 0

# INPUT DETECT
inputCount = 0
INPUT_DETECT_FRAME_THRESHOLD = 30
INPUT_DETECT_FACE_FRAME_THRESHOLD = 10
INPUT_DETECT_INPUTCOUNT_THRESHOLD = 3

def countMouse(e):
    global inputCount
    if isinstance(e, mouse.ButtonEvent) and e.event_type == "up":
        return

    inputCount += 1

def countKeyboard(e):
    global inputCount
    if isinstance(e, keyboard.KeyboardEvent) and e.event_type == "up":
        return

    inputCount += 1

def process(q, vars):
    global curDistanceCheckFrame
    global distanceSum
    global inputCount

    # Detect Input
    curFrameNum = 0
    faceDetectFrameNum = 0
    keyboard.hook(countKeyboard)
    mouse.hook(countMouse)

    alarmDetector = AlarmDetector.AlarmDetecter()
    vars.preEyeTrainTime = time.time()

    trainer.init()
    while vars.isTrain:

        if vars.DISTANCE_MIN_THRESHOLD == None or vars.DISTANCE_MAX_THRESHOLD == None:
            continue

        bgrAndImg = trainer.getWebcamBGRAndGrayImage(480)
        if bgrAndImg == None:
            continue
        bgr, img = bgrAndImg
        #img = cv2.flip(img, 1)

        # test
        #bgr = cv2.flip(bgr, 1)

        rect = trainer.getFaceRect(img)
        if rect != None:
            facialLandmark = trainer.getFacialLandmark(img, rect)

            # ear
            res = alarmDetector.shouldEarAlarmOn(trainer, facialLandmark, earThreshold=vars.earThreshold, BLINK_CHECK_FRAME_NUMBER=BLINK_CHECK_FRAME_NUMBER, BLINK_FRAME_THRESHOLD=BLINK_FRAME_THRESHOLD)
            if res != None:
                q.put(res)


            leftEye = list(facialLandmark.part(i) for i in const.LEFT_EYE_INDEXES)
            leftEye = list((mark.x, mark.y) for mark in leftEye)

            for coor in leftEye:
                cv2.circle(bgr, coor, 2, (0, 0, 255), 1)

            # eye train
            curTime = time.time()
            eyeTrainElapsed = curTime - vars.preEyeTrainTime
            if eyeTrainElapsed > vars.eyeTrainGapSec and not vars.canCheckEyeTrain and not vars.isSendOnEyeTrainMessage:
                q.put([MessageType.EYE_TRAIN_ON, True])
                vars.isSendOnEyeTrainMessage = True

            if vars.canCheckEyeTrain:
                leftEye = np.array(leftEye)
                eyeRect = [
                    (
                        min(leftEye[:, 0]) - const.EYE_MARGIN['x'],
                        min(leftEye[:, 1]) - const.EYE_MARGIN['y']
                    ),
                    (
                        max(leftEye[:, 0]) + const.EYE_MARGIN['x'],
                        max(leftEye[:, 1]) + const.EYE_MARGIN['y']
                    )]
                centers = trainer.getIrisAndEyeCenter(img, eyeRect, vars.eyeTrainThreshold)
                if centers != None:
                    irisCenter, eyeCenter = centers

                    res = alarmDetector.isOpenNextEyeTrain(vars.curEyeTrainPos, irisCenter, eyeCenter, eyeRect)
                    if res != None:
                        vars.canCheckEyeTrain = False
                        q.put(res)

                    for center in centers:
                        tempCenter = (int(center[0]), int(center[1]))
                        cv2.circle(bgr, tempCenter, 2, (0, 255, 255), 1)

            # distance
            rotateMatrix, tvec = trainer.getRotateMatrixAndTransVec(facialLandmark)
            worldCoors = trainer.getObjectsWorldCoors(facialLandmark, [7], rotateMatrix, tvec)
            Coor = worldCoors[0]
            distance = -Coor[2]

            curDistanceCheckFrame += 1
            distanceSum += distance
            if curDistanceCheckFrame >= DISTANCE_CHECK_FRAME_NUMBER:
                distanceMean = distanceSum / DISTANCE_CHECK_FRAME_NUMBER
                if distanceMean < vars.DISTANCE_MIN_THRESHOLD:
                    q.put([MessageType.DISTANCE, DistMessage.INSUFFICIENT])
                elif distanceMean > vars.DISTANCE_MAX_THRESHOLD:
                    q.put([MessageType.DISTANCE, DistMessage.EXCEED])
                else:
                    q.put([MessageType.DISTANCE, DistMessage.OK])
                curDistanceCheckFrame = 0
                distanceSum = 0

        # Input Detect
        if rect != None:
            faceDetectFrameNum += 1
        curFrameNum += 1

        if curFrameNum >= INPUT_DETECT_FRAME_THRESHOLD:
            if faceDetectFrameNum < INPUT_DETECT_FACE_FRAME_THRESHOLD:
                # 얼굴 검출 안되는 자세로 장치 사용
                if inputCount >= INPUT_DETECT_INPUTCOUNT_THRESHOLD:
                    q.put([MessageType.INPUTDETECT, True])
                # 장치 사용 X
                else:
                    q.put([MessageType.EAR, False])
                    q.put([MessageType.DISTANCE, DistMessage.OK])
                    q.put([MessageType.INPUTDETECT, False])
            # 장치 올바르게 사용 중
            else:
                q.put([MessageType.INPUTDETECT, False])

            inputCount = 0
            curFrameNum = 0
            faceDetectFrameNum = 0

        cv2.imshow("image", bgr)
        if cv2.waitKey(1) > 0:
            break

    trainer.dispose()
    keyboard.unhook_all()
    mouse.unhook_all()


distances = []
def reviseDistance(vars, uiQ):
    if vars.isDuringDistanceRevision:
        return

    vars.isDuringDistanceRevision = True

    trainer.init()
    startTime = time.time()
    while time.time() - startTime < 5:
        bgrAndImg = trainer.getWebcamBGRAndGrayImage(480)

        if bgrAndImg == None:
            continue
        bgr, img = bgrAndImg
        rect = trainer.getFaceRect(img)

        cv2.imshow("debug", bgr)
        cv2.waitKey(1)

        if rect == None:
            continue

        facialLandmark = trainer.getFacialLandmark(img, rect)
        rotateMatrix, tvec = trainer.getRotateMatrixAndTransVec(facialLandmark)
        worldCoors = trainer.getObjectsWorldCoors(facialLandmark, [7], rotateMatrix, tvec)
        Coor = worldCoors[0]
        distance = -Coor[2]
        distances.append(distance)

    if len(distances) < 10:
        uiQ.put([UIMessageType.DIST, False])
        vars.DISTANCE_MIN_THRESHOLD = None
        vars.DISTANCE_MAX_THRESHOLD = None
    else:
        distances.sort()
        length = len(distances)
        minDist = distances[int(length * 1/4)]
        maxDist = distances[int(length * 3/4)]
        vars.DISTANCE_MIN_THRESHOLD = minDist - vars.DISTANCE_THRESHOLD_MARGIN
        vars.DISTANCE_MAX_THRESHOLD = maxDist + vars.DISTANCE_THRESHOLD_MARGIN

        uiQ.put([UIMessageType.DIST, True])

    vars.isDuringDistanceRevision = False
    trainer.dispose()

def reviseEyeTrainer(vars):
    trainer.init()
    blobDetectorParams = cv2.SimpleBlobDetector_Params()
    blobDetectorParams.filterByArea = True
    blobDetectorParams.maxArea = 1500
    blobDetector = cv2.SimpleBlobDetector_create(blobDetectorParams)
    
    while vars.isEyeTrainerRevise:

        bgrAndImg = trainer.getWebcamBGRAndGrayImage(480)

        if bgrAndImg == None:
            continue
        bgr, img = bgrAndImg
        rect = trainer.getFaceRect(img)
        if rect == None:
            continue

        facialLandmark = trainer.getFacialLandmark(img, rect)
        leftEye = list(facialLandmark.part(i) for i in const.LEFT_EYE_INDEXES)
        leftEye = list((mark.x, mark.y) for mark in leftEye)
        leftEye = np.array(leftEye)
        eyeRect = [
            (
                min(leftEye[:, 0]) - const.EYE_MARGIN['x'],
                min(leftEye[:, 1]) - const.EYE_MARGIN['y']
            ),
            (
                max(leftEye[:, 0]) + const.EYE_MARGIN['x'],
                max(leftEye[:, 1]) + const.EYE_MARGIN['y']
            )]
        cv2.imshow("test face Image", bgr)
        img = img[eyeRect[0][1]:eyeRect[1][1], eyeRect[0][0]:eyeRect[1][0]]
        bgr = bgr[eyeRect[0][1]:eyeRect[1][1], eyeRect[0][0]:eyeRect[1][0]]


        _, img = cv2.threshold(img, vars.eyeTrainThreshold, 255, cv2.THRESH_BINARY)

        img = cv2.erode(img, None, iterations=1)
        img = cv2.dilate(img, None, iterations=1)
        img = cv2.medianBlur(img, 7)

        blobPoints = blobDetector.detect(img)
        if blobPoints != None and len(blobPoints) != 0:
            irisPoint = blobPoints[0]
            irisPoint = (int(irisPoint.pt[0]), int(irisPoint.pt[1]))
            cv2.circle(bgr, irisPoint, 2, (0, 255, 255), 1)


        showedImg = cv2.resize(bgr, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("revise1", showedImg)
        cv2.imshow("revise2", img)

        cv2.waitKey(1)

    trainer.dispose()

def reviseEar(vars):
    trainer.init()

    fontpath = "./font/NanumSquareR.ttf"
    font = ImageFont.truetype(fontpath, 15)

    isFirstEarCheck = True
    earFrameCount = 0
    preEarFrameCount = 0
    frameCount = 0
    blinkCount = 0
    while vars.isEarRevise:
        bgrAndImg = trainer.getWebcamBGRAndGrayImage(480)

        if bgrAndImg == None:
            continue
        bgr, img = bgrAndImg
        rect = trainer.getFaceRect(img)
        if rect == None:
            continue

        facialLandmark = trainer.getFacialLandmark(img, rect)

        if isFirstEarCheck:
            isFirstEarCheck = False

        ear = trainer.getEyeAspectRatio(facialLandmark)

        if ear < vars.earThreshold:
            earFrameCount += 1

        if frameCount >= BLINK_CHECK_FRAME_NUMBER:
            if earFrameCount >= BLINK_FRAME_THRESHOLD:
                blinkCount += 1

            preEarFrameCount = earFrameCount
            earFrameCount = 0
            frameCount = 0

        frameCount += 1

        textImg = np.full((200, 200, 3), 255, dtype=np.uint8)
        tempTextImg = Image.fromarray(textImg)
        draw = ImageDraw.Draw(tempTextImg)
        draw.text((0, 75), f"눈 깜빡임 수 : {blinkCount}", font=font, fill=(0, 0, 0, 0))
        draw.text((0, 126), f"Pre Ear Frame Count : {preEarFrameCount} >= {BLINK_FRAME_THRESHOLD}", font=font, fill=(0, 0, 0, 0))
        textImg = np.array(tempTextImg)


        cv2.imshow("debug", bgr)
        cv2.imshow("revise", textImg)
        cv2.waitKey(1)

    trainer.dispose()


class MessageType(Enum):
    EAR = 1
    EYE_TRAIN = 2
    EYE_TRAIN_ON = 3
    DISTANCE = 4
    INPUTDETECT = 5

class UIMessageType(Enum):
    DIST = 1
    EYETRAIN_IMG_SHOW = 2

class DistMessage(Enum):
    EXCEED = 1
    INSUFFICIENT = 2
    OK = 3
