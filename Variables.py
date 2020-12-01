import win32api


class Variables:
    _instance = None

    @classmethod
    def setVariable(cls, var):
        var.curEyeTrainPos = None
        var.canCheckEyeTrain = False
        var.eyeTrainGapSec = 60 * 10
        var.preEyeTrainTime = None
        var.isSendOnEyeTrainMessage = False
        var.isTrain = False
        var.isDuringDistanceRevision = False

        #DISTNACE
        var.DISTANCE_MIN_THRESHOLD = None
        var.DISTANCE_MAX_THRESHOLD = None
        var.DISTANCE_THRESHOLD_MARGIN = 1.5 # todo main UI Progress Bar

        #EYE TRAIN
        var.eyeTrainThreshold = 50

        #EAR
        var.earThreshold = 30

        var.isEyeTrainerRevise = False
        var.isEarRevise = False

        return var

    @classmethod
    def setEyeTrainPos(self, var, widgetX, widgetY):
        screenHeight = win32api.GetSystemMetrics(1)
        screenWidth = win32api.GetSystemMetrics(0)

        x = widgetX + screenWidth / 2
        y = -widgetY + screenHeight / 2

        var.curEyeTrainPos = [x, y]


