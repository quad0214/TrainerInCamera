import cv2
import dlib
import Constants
import numpy as np
import json

class Trainer:
    def __init__(self) :
        self.cap = None
        self.faceDectector = dlib.get_frontal_face_detector()
        self.facialLandmarkDetector = dlib.shape_predictor(Constants.FACIAL_LANDMARK_DETECTOR)
        self.eyeDetector = cv2.CascadeClassifier(Constants.HAARCASCADE_EYE_DETECTOR)

        blobDetectorParams = cv2.SimpleBlobDetector_Params()
        blobDetectorParams.filterByArea = True
        blobDetectorParams.maxArea = 1500
        self.blobDetector = cv2.SimpleBlobDetector_create(blobDetectorParams)

    #return width * heigth list
    def getWebcamBGRAndGrayImage(self, width):
        ret, frame = self.cap.read()
        if ret == False:
            return None

        imgWidth, imgHeight, _ = frame.shape
        multiplier = width / imgWidth
        height = int(imgHeight * multiplier)


        frame = cv2.resize(frame, (height, width))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, gray

    # return face rect(rectangle type - [(left, top), (right, bottom)])
    # if dosent exist return None
    def getFaceRect(self, img):
        faceRects = self.faceDectector(img, 0)

        if len(faceRects) == 0:
            return None

        faceRect = faceRects.pop()
        for rect in faceRects:
            if faceRect.width < rect:
                faceRect = rect

        return faceRect

    # return facial landmark(type : full_object_detection
    # http://dlib.net/python/index.html?highlight=full_object_detection#dlib.full_object_detection
    # )
    def getFacialLandmark(self, img, faceRect):
        facialLandmark = self.facialLandmarkDetector(img, faceRect)
        return facialLandmark

    def getEyeAspectRatio(self, facialLanmark):
        poses = []
        for leftEyeIndex in Constants.LEFT_EYE_INDEXES:
            poses.append(facialLanmark.part(leftEyeIndex))
        poses = list((pose.x, pose.y) for pose in poses)
        poses = np.array(poses)
        poses = poses.astype(float)

        height1Vec = poses[1] - poses[5]
        height1 = np.abs(np.dot(height1Vec, height1Vec))
        height2Vec = poses[2] - poses[4]
        height2 = np.abs(np.dot(height2Vec, height2Vec))
        widthVector = poses[0] - poses[3]
        width = np.abs(np.dot(widthVector, widthVector))
        ear = height1 + height2 / (2*width)

        return ear

    def getIrisAndEyeCenter(self, img, eyeRect, eyeTrainThreshold):
        img = img[eyeRect[0][1]:eyeRect[1][1], eyeRect[0][0]:eyeRect[1][0]]

        eyeCenterPos = ((eyeRect[0][0] + eyeRect[1][0]) // 2, (eyeRect[0][1] + eyeRect[1][1]) // 2)

        # for test
        showedImg = cv2.resize(img, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("eye", showedImg)

        _, img = cv2.threshold(img, eyeTrainThreshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("thresholding", img)

        img = cv2.erode(img, None, iterations=1)
        img = cv2.dilate(img, None, iterations=1)
        img = cv2.medianBlur(img, 7)
        cv2.imshow("imgProcessing", img)

        blobPoints = self.blobDetector.detect(img)
        if blobPoints is None or len(blobPoints) == 0:
            return None

        irisPoint = blobPoints[0]
        irisCenterPos = ((eyeRect[0][0] + irisPoint.pt[0]), (eyeRect[0][1] + irisPoint.pt[1]))

        return (irisCenterPos, eyeCenterPos)

    def getRotateMatrixAndTransVec(self, facialLandmark):
        imagePoses = list((facialLandmark.part(index)) for index in Constants.DISTANCE_PIVOT_INDEXES)
        imagePoses = list((part.x, part.y) for part in imagePoses)

        _, rvec, tvec = cv2.solvePnP(np.array(Constants.DISTANCE_PIVOT_POSES, dtype='float32'), np.array(imagePoses, dtype="float32"), np.array(Constants.CAMERA_MATRIX, dtype="float32"), np.zeros((8, 1), dtype="float32"))

        rotateMatrix, _ = cv2.Rodrigues(rvec)

        return (rotateMatrix, tvec)

    def getObjectsWorldCoors(self, facialLandmark, objectIndexes, rotateMatrix, tvec):
        rotateMatrixInv = np.linalg.inv(rotateMatrix)
        cameraMatrixInv = np.linalg.inv(Constants.CAMERA_MATRIX)
        sampleImageCoor = facialLandmark.part(Constants.DISTANCE_PIVOT_INDEXES[0])
        sampleImageCoor = [sampleImageCoor.x, sampleImageCoor.y, 1]
        sampleImageCoor = np.array(sampleImageCoor, dtype="float32")
        sampleImageCoor = sampleImageCoor.T
        sampleWorldCoor = np.array(Constants.DISTANCE_PIVOT_POSES[0], dtype="float32")

        #코로 좌표계가 고정되어 버려서 코를 기준점으로 하면 안됨.
        # 움직이지 않는 것으로 고정하거나 혹은 scaleFactor를 가정해서 정하셈
        # scaleFactor = np.dot(rotateMatrixInv, tvec)[2] + sampleWorldCoor[2]
        # scaleFactor /= np.dot(np.dot(rotateMatrixInv, cameraMatrixInv)[2, :], sampleImageCoor)
        # print("scaleFactor", scaleFactor)

        objectsCoors = []
        for objectIndex in objectIndexes:
            scaleFactor = 1
            objectPart = facialLandmark.part(objectIndex)
            objectImageCoor = np.array([objectPart.x, objectPart.y, 1])
            objectImageCoor = objectImageCoor.T

            objectWorldCoor = np.dot(rotateMatrixInv, cameraMatrixInv) * scaleFactor
            objectWorldCoor = np.dot(objectWorldCoor, objectImageCoor)
            objectWorldCoor.resize((3, 1))
            objectWorldCoor = np.subtract(objectWorldCoor, np.dot(rotateMatrixInv, tvec))

            objectsCoors.append(objectWorldCoor)

        return objectsCoors

    def getFaceDirection(self, rotateMatrix, tVec):
        rotateMatrix = np.array(rotateMatrix)
        tVec = np.array(tVec).T
        projectionMatrix = np.concatenate((rotateMatrix, tVec), axis=1)
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(projectionMatrix)
        print(eulerAngles)

    def dispose(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def init(self):
        self.cap = cv2.VideoCapture(0)