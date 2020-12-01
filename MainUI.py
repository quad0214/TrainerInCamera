import sys
import UIConstants as Const
from TrainerProcess import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import win32api
from multiprocessing import Process, Queue, Manager
from Variables import Variables
import pyglet.window
from ctypes import windll
import numpy


class Consumer(QThread):
    poped = pyqtSignal(list)

    def __init__(self, q):
        super().__init__()
        self.q = q


    def run(self):
        while True:
            if not self.q.empty():
                data = self.q.get()
                self.poped.emit(data)

class UIConsumer(QThread):
    poped = pyqtSignal(list)

    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        while True:
            if not self.q.empty():
                data = self.q.get()
                self.poped.emit(data)


class MainWindow(QWidget):
    def __init__(self, q):
        super().__init__()

        # consumer thread start -> get trainer info
        self.consumer = Consumer(q)
        self.consumer.poped.connect(self.processMessage)
        self.consumer.start()

        self.uiConsumer = UIConsumer(uiQ)
        self.uiConsumer.poped.connect(self.processUIMessage)
        self.uiConsumer.start()

        self.vars = sharedVariables

        self.setupUI()

        self.isOpenDistNotification = False
        self.isOpenEyeTrain = False
        self.notification = NotificationWidget()
        self.eyeTrainer = EyeTrainWidget()

        self.eyeTrainer.init(6, 1)

        self.isDuringDistanceRevise = False
        self.isDistanceRevise = False

        self.blueLightWindow = pyglet.window.Window(visible=False)
        self.oriRamps = np.empty((3, 256), dtype=np.uint16)
        windll.gdi32.GetDeviceGammaRamp(self.blueLightWindow._dc, self.oriRamps.ctypes)


    def setupUI(self):
        self.setWindowTitle("카메라 속 트레이너")
        self.setGeometry(200, 200, 300, 500)

        distLabel = QLabel("거리 보정 버튼을 누른 뒤, 3초 동안 올바른 자세를 취해주세요.", self)
        distLabel.setWordWrap(True)
        self.distButton = QPushButton("거리 보정 시작", self)
        self.distButton.clicked.connect(self.startDistanceRevise)
        self.distDetail = QLabel("보정 안됨", self)
        self.distDetail.setWordWrap(True)

        eyeTrainerLabel = QLabel("안구 운동 보정 버튼을 누른 뒤, 새로운 창에 홍채만 남을 때까지 slider를 조절해주세요", self)
        eyeTrainerLabel.setWordWrap(True)
        self.eyeTrainerSlider = QSlider(Qt.Horizontal, self)
        self.eyeTrainerSlider.setRange(0, 300)
        self.eyeTrainerSlider.setValue(self.vars.eyeTrainThreshold)
        self.eyeTrainerSlider.sliderMoved.connect(self.setEyeTrainThreshold)
        self.eyeTrainerSlider.setEnabled(False)
        self.eyeTrainerButton = QPushButton("안구 운동 보정 시작", self)
        self.eyeTrainerButton.clicked.connect(self.startEyeTrainerRevise)
        self.eyeTrainerDetail = QLabel(f"보정 안됨. default threshold : {self.vars.eyeTrainThreshold}", self)
        self.eyeTrainerDetail.setWordWrap(True)

        earLabel = QLabel("눈 깜빡임 보정")
        self.earSlider = QSlider(Qt.Horizontal, self)
        self.earSlider.setRange(0, 100)
        self.earSlider.setValue(self.vars.earThreshold)
        self.earSlider.sliderMoved.connect(self.setEarThreshold)
        self.earSlider.setEnabled(False)
        self.earButton = QPushButton("눈 깜빡임 보정 시작", self)
        self.earButton.clicked.connect(self.startEarRevise)
        self.earDetail = QLabel(f"보정 안됨. default threshold : {self.vars.earThreshold}", self)
        self.earDetail.setWordWrap(True)

        self.debugStartEyeTrainButton = QPushButton("안구 운동 시작(디버그)", self)
        self.debugStartEyeTrainButton.clicked.connect(self.startEyeTrain)
        self.debugStartEyeTrainButton.setEnabled(False)

        self.startButton = QPushButton("트레이너 시작!", self)
        self.startButton.clicked.connect(self.startTrain)
        self.startButton.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(distLabel)
        layout.addWidget(self.distButton)
        layout.addWidget(self.distDetail)
        layout.addStretch(1)
        layout.addWidget(eyeTrainerLabel)
        layout.addWidget(self.eyeTrainerSlider)
        layout.addWidget(self.eyeTrainerButton)
        layout.addWidget(self.eyeTrainerDetail)
        layout.addStretch(1)
        layout.addWidget(earLabel)
        layout.addWidget(self.earSlider)
        layout.addWidget(self.earButton)
        layout.addWidget(self.earDetail)
        layout.addStretch(3)
        layout.addWidget(self.debugStartEyeTrainButton)
        layout.addStretch(1)
        layout.addWidget(self.startButton)

        self.setLayout(layout)
        self.show()

    def startDistanceRevise(self):
        self.distButton.setEnabled(False)
        self.eyeTrainerButton.setEnabled(False)
        self.earButton.setEnabled(False)
        self.startButton.setEnabled(False)
        p = Process(name="DistanceReviseProcess", target=reviseDistance, args=(sharedVariables, uiQ ), daemon=True)
        p.start()

    def startEyeTrainerRevise(self):
        self.eyeTrainerButton.clicked.disconnect()
        self.eyeTrainerButton.clicked.connect(self.stopEyeTrainerRevise)
        self.eyeTrainerSlider.setEnabled(True)
        self.eyeTrainerButton.setText("안구 운동 보정 종료")

        self.distButton.setEnabled(False)
        self.startButton.setEnabled(False)
        self.earButton.setEnabled(False)

        self.vars.isEyeTrainerRevise = True
        p = Process(name="EyeTrainerReviseProcess", target=reviseEyeTrainer, args=(sharedVariables, ), daemon=True)
        p.start()

    def stopEyeTrainerRevise(self):
        self.eyeTrainerButton.clicked.disconnect()
        self.eyeTrainerButton.clicked.connect(self.startEyeTrainerRevise)
        self.eyeTrainerSlider.setEnabled(False)
        self.eyeTrainerDetail.setText(f"threshold : {self.vars.eyeTrainThreshold}")
        self.eyeTrainerButton.setText("안구 운동 보정 시작")

        self.distButton.setEnabled(True)
        self.earButton.setEnabled(True)

        self.enableStartButtonPossible()

        self.vars.isEyeTrainerRevise = False

    def startEarRevise(self):
        self.earButton.clicked.disconnect()
        self.earButton.clicked.connect(self.stopEarRevise)
        self.earButton.setText("눈 깜빡임 보정 종료")
        self.earSlider.setEnabled(True)

        self.distButton.setEnabled(False)
        self.startButton.setEnabled(False)
        self.eyeTrainerButton.setEnabled(False)

        self.vars.isEarRevise = True
        p = Process(name="EarReviseProcess", target=reviseEar, args=(sharedVariables, ), daemon=True)
        p.start()

    def stopEarRevise(self):
        self.earButton.clicked.disconnect()
        self.earButton.clicked.connect(self.startEarRevise)
        self.earSlider.setEnabled(False)
        self.earButton.setText("눈 깜빡임 보정 시작")
        self.earDetail.setText(f"threshold : {self.vars.earThreshold}")

        self.distButton.setEnabled(True)
        self.eyeTrainerButton.setEnabled(True)
        self.enableStartButtonPossible()

        self.vars.isEarRevise = False

    def setEyeTrainThreshold(self):
        self.vars.eyeTrainThreshold = self.eyeTrainerSlider.value()

    def setEarThreshold(self):
        self.vars.earThreshold = self.earSlider.value()

    def startEyeTrain(self):
        if not self.vars.isTrain:
            return
        self.eyeTrainer.init(6, 1)
        self.eyeTrainer.on()

    def startTrain(self):
        if not self.isDistanceRevise:
            return

        self.distButton.setEnabled(False)
        self.eyeTrainerButton.setEnabled(False)
        self.earButton.setEnabled(False)
        self.debugStartEyeTrainButton.setEnabled(True)

        self.startButton.clicked.disconnect()
        self.startButton.clicked.connect(self.stopTrain)
        self.startButton.setText("트레이너 종료!")

        self.vars.isTrain = True

        self.startBluelightFilter()

        p = Process(name="TrainerProcess", target=process, args=(q, sharedVariables), daemon=True)
        p.start()

    def stopTrain(self):
        self.vars.isTrain = False

        self.distButton.setEnabled(True)
        self.eyeTrainerButton.setEnabled(True)
        self.earButton.setEnabled(True)
        self.debugStartEyeTrainButton.setEnabled(False)

        self.startButton.clicked.disconnect()
        self.startButton.clicked.connect(self.startTrain)
        self.startButton.setText("트레이너 시작!")

        self.stopBluelightFilter()

        self.notification.hide()
        self.eyeTrainer.hide()

    def startBluelightFilter(self):
        startScale = np.array([[1], [1], [1]], dtype="float32")
        scale = np.array([[1], [0.8], [0.8]], dtype="float32")

        step = 40
        for i in range(0, step + 2):
            scaleIndex = float(i) / (step + 1)
            currentScale = np.multiply(startScale, (1 - scaleIndex) + np.multiply(scale, scaleIndex))
            curRamps = np.uint16(np.round(np.multiply(currentScale, self.oriRamps)))

            windll.gdi32.SetDeviceGammaRamp(self.blueLightWindow._dc, curRamps.ctypes)

        newRamps = np.uint16(np.round(np.multiply(scale, self.oriRamps)))
        windll.gdi32.SetDeviceGammaRamp(self.blueLightWindow._dc, newRamps.ctypes)

    def stopBluelightFilter(self):
        windll.gdi32.SetDeviceGammaRamp(self.blueLightWindow._dc, self.oriRamps.ctypes)

    @pyqtSlot(list)
    def processMessage(self, dataList):
        dataType = dataList[0]
        if dataType == MessageType.EAR:
            isOpenNotification = dataList[1]
            if isOpenNotification:
                self.notification.on(isEar=True, isDistance=False, isInputDetect=False)
            else:
                self.notification.off(isEar=True, isDistance=False, isInputDetect=False)

        elif dataType == MessageType.EYE_TRAIN:
            isOpenNextEyeTrain = dataList[1]
            if isOpenNextEyeTrain:
                isProgress = self.eyeTrainer.showNextAlarm()
                if isProgress:
                    self.vars.canCheckEyeTrain = True
                else:
                    print("isNotProgress")
                    self.vars.canCheckEyeTrain = False
                self.vars.preEyeTrainTime = time.time()

        elif dataType == MessageType.EYE_TRAIN_ON:
            isOpenEyeTrain = dataList[1]
            if isOpenEyeTrain:
                self.eyeTrainer.on()

        elif dataType == MessageType.DISTANCE:
            distMessage = dataList[1]
            isOpenNotification = False
            if distMessage != DistMessage.OK:
                isOpenNotification = True
            if isOpenNotification :
                self.notification.on(isEar=False, isDistance=True, distMessage=distMessage, isInputDetect=False)
            else:
                self.notification.off(isEar=False, isDistance=True, isInputDetect=False)

        elif dataType == MessageType.INPUTDETECT:
            isOpenNotification = dataList[1]
            if isOpenNotification:
                self.notification.on(isEar=False, isDistance=False, isInputDetect=True)
            else:
                self.notification.off(isEar=False, isDistance=False, isInputDetect=True)


    @pyqtSlot(list)
    def processUIMessage(self, dataList):
        type = dataList[0]
        if type == UIMessageType.DIST:
            isSuccess = dataList[1]
            if isSuccess:
                self.distDetail.setText(f"보정 완료. 설정된 거리 - 최소 거리 : {self.vars.DISTANCE_MIN_THRESHOLD}, 최대 거리 : {self.vars.DISTANCE_MAX_THRESHOLD}")
                self.isDistanceRevise = True
            else:
                self.distDetail.setText("보정 실패. 다시 시도해주세요.")
                self.isDistanceRevise = False

            self.distButton.setEnabled(True)
            self.eyeTrainerButton.setEnabled(True)
            self.earButton.setEnabled(True)
            self.enableStartButtonPossible()

        self.enableStartButtonPossible()

    def enableStartButtonPossible(self):
        #todo differenct Revise Add
        if self.isDistanceRevise:
            self.startButton.setEnabled(True)

    def closeEvent(self, e):
        if self.vars.isTrain:
            self.stopTrain()

        self.notification.deleteLater()
        self.eyeTrainer.deleteLater()
        self.blueLightWindow.close()

        self.deleteLater()

class EyeTrainWidget(QWidget):
    def __init__(self):
        self.alarmStep = None
        self.cycleNumber = None
        self.curAlarmIndex = 0
        self.curCycleIndex = 0
        self.isFirst = True
        self.alarmWidgets = []
        self.vars = sharedVariables

        super().__init__()
        self.setup()

    # todo 알림 실행 중 init할 수도 있음 -> isFirst로 검사
    def init(self, alarmNumber, cycleNumber):
        if alarmNumber < 0 or alarmNumber > len(Const.EyeTrainCoorRatios):
            raise Exception("alarmNumber 범위 벗어남")
        self.alarmStep = alarmNumber // len(Const.EyeTrainCoorRatios)
        self.cycleNumber = cycleNumber
        self.isProcessing = False


    def on(self):
        print("ON")

        self.off()

        self.show()
        curAlarmWidget = self.alarmWidgets[self.curAlarmIndex]
        curAlarmWidget.show()
        Variables.setEyeTrainPos(self.vars, curAlarmWidget.x(), curAlarmWidget.y())
        self.vars.canCheckEyeTrain = True
        self.vars.isSendOnEyeTrainMessage = False
        self.isProcessing = True

    def off(self):
        print("OFF")
        self.curCycleIndex = 0
        self.curAlarmIndex = 0
        self.vars.canCheckEyeTrain = False
        self.vars.isSendOnEyeTrainMessage = False
        self.vars.preEyeTrainTime = time.time()
        self.isProcessing = False

        self.hide()

    def setup(self):
            self.setupUI()
            self.setupFlags()
            self.show()

    def setupFlags(self):
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.WindowTransparentForInput
            )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def setupUI(self):
        # setup alarm widget
        alarmImagePixmap = QPixmap()
        alarmImagePixmap.load("./img/circle.png")
        alarmImagePixmap = alarmImagePixmap.scaled(32, 32)
        imgSize = alarmImagePixmap.size()

        for ratio in Const.EyeTrainCoorRatios:
            y = ratio[0] * win32api.GetSystemMetrics(1) - (imgSize.height()//2)
            x = ratio[1] * win32api.GetSystemMetrics(0) - (imgSize.width()//2)

            widget = QLabel(f"{y} {x}", self)
            widget.move(int(x), int(y))
            widget.setPixmap(alarmImagePixmap)
            self.alarmWidgets.append(widget)
            widget.hide()

        # setup main window
        screenHeight = win32api.GetSystemMetrics(1)
        screenWidth = win32api.GetSystemMetrics(0)
        self.setGeometry(0, 0, screenWidth, screenHeight)

        # test
        # button = QPushButton("button", self)
        # button.setCheckable(True)
        # button.pressed.connect(self.showNextAlarm)


    def showNextAlarm(self):
        print(self.curAlarmIndex)
        self.alarmWidgets[self.curAlarmIndex].hide()
        self.curAlarmIndex += self.alarmStep
        if self.curAlarmIndex >= len(Const.EyeTrainCoorRatios):
            self.curAlarmIndex = 0
            self.curCycleIndex += 1

        if self.curCycleIndex >= self.cycleNumber:
            return False

        curAlarmWidget = self.alarmWidgets[self.curAlarmIndex]
        curAlarmWidget.show()
        Variables.setEyeTrainPos(self.vars, curAlarmWidget.x(), curAlarmWidget.y())
        return True

class NotificationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.isOpenedEar = False
        self.isOpenedDistance = False
        self.isOpenedInputDetect = False
        self.setup()
        self.distMessage = None

    def on(self, isEar, isDistance, isInputDetect, distMessage=None):
        text = ""
        if isEar or self.isOpenedEar:
            self.isOpenedEar = True
            text += Const.EAR_NOTIFICATION_TEXT
            text += "\n"

        if isDistance:
            self.isOpenedDistance = True
            self.distMessage = distMessage
        if self.isOpenedDistance:
            if self.distMessage == DistMessage.EXCEED:
                text += Const.DISTANCE_NOTIFICATION_EXCEED_TEXT
            elif self.distMessage == DistMessage.INSUFFICIENT:
                text += Const.DISTANCE_NOTIFICATION_INSUFFICIENT_TEXT
            text += "\n"

        if isInputDetect or self.isOpenedInputDetect:
            self.isOpenedInputDetect = True
            text += Const.INPUT_DETECT_NOTIFICATION_TEXT

        self.messageLabel.setText(text)

        self.show()

    def off(self, isEar, isDistance, isInputDetect):
        if isEar:
            self.isOpenedEar = False
        if isDistance:
            self.distMessage = None
            self.isOpenedDistance = False
        if isInputDetect:
            self.isOpenedInputDetect = False

        text = ""
        if self.isOpenedEar:
            text += Const.EAR_NOTIFICATION_TEXT
            text += "\n"
        if self.isOpenedDistance:
            if self.distMessage == DistMessage.EXCEED:
                text += Const.DISTANCE_NOTIFICATION_EXCEED_TEXT
            elif self.distMessage == DistMessage.INSUFFICIENT:
                text += Const.DISTANCE_NOTIFICATION_INSUFFICIENT_TEXT
            text += "\n"
        if self.isOpenedInputDetect:
            text += Const.INPUT_DETECT_NOTIFICATION_TEXT

        if text == "":
            self.hide()
        else:
            self.messageLabel.setText(text)
            self.show()


    def setup(self):
        self.setupUI()
        self.setupFlags()

    def setupUI(self):
        self.messageLabel = QLabel("None", self)
        self.messageLabel.setWordWrap(True)

        # message label size & pos
        screenHeight = win32api.GetSystemMetrics(1)
        screenWidth = win32api.GetSystemMetrics(0)
        height = screenHeight // 5
        width = screenWidth // 5
        y = screenHeight // 2 - height // 2
        x = screenWidth // 2 - width // 2
        self.messageLabel.setGeometry(x, y, width, height)

        # message label background
        self.messageLabel.setAutoFillBackground(True)
        self.messageLabel.setStyleSheet("background-color: rgba(255, 255, 255, 0.7)")

        # message font
        labelFont = self.messageLabel.font()
        labelFont.setPointSize(15)
        labelFont.setBold(True)
        self.messageLabel.setFont(labelFont)
        self.messageLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # main widget
        self.setGeometry(0, 0, screenWidth, screenHeight)

        # todo test
        # button = QPushButton("exit", self)
        # button.clicked.connect(self.remove)

    def setupFlags(self):
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

if __name__ == "__main__":
    print("Start INIT")
    q = Queue()
    uiQ = Queue()

    manager = Manager()
    sharedVariables = manager.Namespace()
    sharedVariables = Variables.setVariable(sharedVariables)

    print("Finish INIT")
    app = QApplication(sys.argv)
    mainWindow = MainWindow(q)

    #eyeTrainWindow = EyeTrainWidget(8, 1)
    #notificationWidget = NotificationWidget("아아아ㅏ아아")

    sys.exit(app.exec_())


