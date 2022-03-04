import tkinter.filedialog

from CNN.ModelUtility.go import PNG_JPG
from GUI.QtDesignerFile.singleImgClassificationWin import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PIL import Image
import cv2 as cv
import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
# from predict import predict
# from CNN.ModelUtility.test import predict
from CNN.ModelUtility.singleImgEval import predict


class UiMain(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.setupUi(self)
        self.fileBtn.clicked.connect(self.loadImage)
        self.fileBtn2.clicked.connect(self.Predictedresults)
        self.i = ''

    # 打开文件功能
    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片', '.', '图像文件(*.jpg *.png)')
        if self.fname:

            print(self.fname.replace("\\", "/"))
            if self.fname.endswith('.png') or self.fname.endswith('.PNG'):
                PNG_JPG(self.fname.replace("\\", "/"))
                self.fname = (self.fname[:-3] + "jpg")
                print(self.fname)
            self.Infolabel.setText("Open successfully\n" + self.fname)
            jpg = QtGui.QPixmap(self.fname).scaled(self.Imglabel.width(), self.Imglabel.height())
            self.i = True

            self.Imglabel.setPixmap(jpg)
            result = predict(self.fname)
            self.Infolabel.setText(result[0])

        else:
            self.Infolabel.setText("Please select an image.")

    def otherCilck(self):
        # self.app = QApplication(sys.argv)
        self.ui = UiMain()
        self.ui.show()
        # self.sys.exit(app.exec_())

    def Predictedresults(self):
        if self.i:
            root = tkinter.Tk()
            root.withdraw()
            result = predict(self.fname)
            tkinter.messagebox.showinfo('Successful', result[1])
        else:
            root = tkinter.Tk()
            root.withdraw()
            tkinter.messagebox.showinfo('Notification', 'Please select an image first!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
