import os.path
import xlwt
from GUI.QtDesignerFile.multiImgClassificationWin import Ui_Dialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from CNN.ModelUtility.multiImgEval import predict


class UiMain(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.setupUi(self)
        self.fileBtn.clicked.connect(self.loadImage)
        self.result = ''
        self.save.clicked.connect(self.savetxt)
        self.Excel.clicked.connect(self.saveExcel)
        self.PATH = os.path.abspath(r'../../prediction_result')
        if not os.path.exists(self.PATH): os.mkdir(path=self.PATH)

    # 打开文件功能
    def loadImage(self):
        self.fname = QtWidgets.QFileDialog.getExistingDirectory(None, "Please select a folder.")  # 起始路径
        if self.fname:
            print(self.fname)
            self.textEdit.setFontPointSize(30)
            self.textEdit.insertPlainText("Open folder successfully!\n-----------------------\n" + self.fname+'\n-----------------------\n')
            self.result = predict(self.fname)
            print(self.result)
            self.textEdit.insertPlainText(self.result)


        else:
            # print("打开文件失败")
            self.textEdit.insertPlainText("Please select a folder!")

    def savetxt(self):
        if self.result:
            path = os.path.join(self.PATH, 'classification_result.txt')
            f = open(path, mode='w')
            f.write(self.result)
            list1 = self.result.split()

            self.textEdit.clear()
            self.textEdit.insertPlainText('File saved to TXT successfully! You have saved the file to the current folder path')
            print(list1)

        else:
            self.textEdit.insertPlainText("Please make a prediction first!")

    def saveExcel(self):
        if self.result:
            list1 = self.result.split()
            a = 0
            b = 0
            w = xlwt.Workbook()
            ws = w.add_sheet('go')
            for i in list1:
                ws.write(a // 3, b % 3, i)
                print(a // 2)
                print(b % 3)
                print(i)
                a += 1
                b += 1
            path = os.path.join(self.PATH, 'classification_result.xls')
            w.save(path)
            self.textEdit.clear()
            self.textEdit.insertPlainText('File saved as Excel successfully! You have saved the file to the current folder path')
        else:
            self.textEdit.insertPlainText("Please make a prediction first!")

    def otherCilck(self):
        # self.app = QApplication(sys.argv)
        self.ui = UiMain()
        self.ui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
