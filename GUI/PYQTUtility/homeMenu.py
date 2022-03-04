from GUI.QtDesignerFile.mainWin import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
import singleImgClassification
import multiImgClassification

class UiMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.go = singleImgClassification.UiMain()
        self.go2 = multiImgClassification.UiMain()
        self.setupUi(self)
        # self.fileBtn.clicked.connect(self.loadImage)
        path = 'backgroundImg.jpg'
        jpg = QtGui.QPixmap(path).scaled(self.Imglabel.width(), self.Imglabel.height())
        self.Imglabel.setPixmap(jpg)
        self.push.clicked.connect(lambda: ui.showohter())
        self.pushButton_2.clicked.connect(lambda: ui.showohter2())

    def close_w1(self):
        self.close()

    def showohter(self):
        self.go.otherCilck()
        # self.close_w1()

    def showohter2(self):
        self.go2.otherCilck()
        # self.close_w1()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
