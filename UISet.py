# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:14:07 2020

@author: han
"""
from PyQt5.Qt import *
import numpy as np
import sys
from CNNFruit import CNN_init,run_CNN,text_Accuracy,batch_process
import skimage
import cv2 as cv
import os
from csv_fruitinfo import *
import image_algorithm



class btn(QPushButton):
    def enterEvent(self, *args, **kwargs):
        self.setStyleSheet("background-color: yellow")
    def leaveEvent(self, *args, **kwargs):
        self.setStyleSheet("background-color: white")

class MyWindow(QWidget):

    def init_variables(self,*args):
        self.x1 = args[0]['x']
        self.keep_prob = args[0]['keep_prob']
        self.prediction = args[0]['prediction']
        self.n_classes = args[0]['n_classes']
        self.sess = args[1]
        self.list_batch = []
        self.list_name = []

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.init_variables(*args)
        self.set_ui()

    def single_classification(self):

        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")  # 文件夹路径
        if len(directory1[0]) > 1:
            pix = QPixmap(directory1[0])
            pix = pix.scaled(QSize(200, 200))
            self.lb_old.setPixmap(pix)
            img = skimage.data.imread(directory1[0])
            if img.shape != (100, 100, 3):
                print(img.shape)
                img = cv.resize(img, (100, 100))
            img = img[np.newaxis, :, :, :]
            fruit_name = run_CNN(self.x1, self.keep_prob, self.prediction, img, self.sess)
            self.lb_text.setText('This is a ' + fruit_name)
        else:
            return

    def window_init(self):
        self.setWindowTitle("FruitRecognition")
        self.setFixedSize(500, 500)
        self.setWindowIcon(QIcon('icon/icon.jpg'))

    def action_accuracy(self):

        Accuracy = text_Accuracy(self.x1, self.keep_prob, self.prediction, self.sess, self.n_classes)
        self.lb_text.setText('Accuracy:' + str(Accuracy))

    def map_label(self):
        self.lb_old = QLabel(self)
        self.lb_old.move(25, 30)
        self.lb_old.resize(200, 200)
        self.lb_old.setStyleSheet(
            "border-width: 2px;border-style: solid;border-color: rgb(125, 125, 125);")

        self.lb_new = QLabel(self)
        self.lb_new.move(275, 30)
        self.lb_new.resize(200, 200)
        self.lb_new.setStyleSheet(
            "border-width: 2px;border-style: solid;border-color: rgb(125, 125, 125);")

        self.lb_text = QLabel(self)
        self.lb_text.move(20, 250)
        self.lb_text.resize(450, 220)
        self.lb_text.setStyleSheet(
            "font:20px '楷体';border-width: 2px;border-style: solid;border-color: rgb(125, 125, 125);")
        self.lb_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def batch_classification(self):

        def is_dir(path,i):
            if os.path.isdir(path) == False:
                if image_algorithm.is_img(i):
                    img_data = skimage.data.imread(path)
                    if img_data.shape != (100, 100, 3):
                        img_data = cv.resize(img_data, (100, 100))
                    self.list_batch.append(img_data)
                    self.list_name.append(i)
                else:
                    return
            else:
                for i in os.listdir(path):
                    path_file = path + '/' + i
                    is_dir(path_file,i)

        self.list_batch.clear()
        self.list_name.clear()
        PathFolder = QFileDialog.getExistingDirectory()
        img = os.listdir(PathFolder)
        for i in img:
            Path = PathFolder + '/' + i
            is_dir(Path, i)
        batch_label = batch_process(self.x1, self.keep_prob, self.prediction, self.sess, self.list_batch)
        list_name = np.array(self.list_name).T
        batch_label = np.array(batch_label).T
        fruit_data = np.column_stack((list_name,batch_label))
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "提示",
                                        "请选择保存文件的路径",
                                        QMessageBox.Yes)

        fileName_choose=QFileDialog.getSaveFileName(filter="csv (*.csv);;All Files (*)")
        if len(fileName_choose[0])<1:
            return
        save_fruitinfo(fileName_choose[0], fruit_data)
        QMessageBox.information(self,  "提示", "保存文件完成")

    def gray_segmentation(self):

        text, okPressed = QInputDialog.getText(self, "输入框", "请输入分割的数量:", QLineEdit.Normal, "")

        if okPressed == False:
            return
        elif 2 > int(text) or int(text) > 9:
            reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "提示",
                                        "请在LineEdit中输入2-9",
                                        QMessageBox.Yes)
        else:
            directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")  # 文件夹路径
            if len(directory1[0]) > 1:
                pix = QPixmap(directory1[0])
                pix = pix.scaled(QSize(200, 200))
                self.lb_old.setPixmap(pix)
                reply = QMessageBox.information(self,  # 使用infomation信息框
                                                "提示",
                                                "请选择保存文件的路径",
                                                QMessageBox.Yes)

                fileName_choose = QFileDialog.getExistingDirectory()
                if len(fileName_choose) < 1:
                    return
                image_path = image_algorithm.kmean_img(directory1[0], fileName_choose, n_clusters=int(text))
                print(image_path)
                pix = QPixmap(image_path)
                pix = pix.scaled(QSize(200, 200))
                self.lb_new.setPixmap(pix)
                QMessageBox.information(self, "提示", "保存文件完成")
                self.lb_text.setText("分割成功")
            else:
                return

    def color_segmentation(self):
        state_list = []

        directory1 = QFileDialog.getOpenFileName(self, "选择图片文件", "H:/", "Images(*.png *.jpg *.jpeg *.bmp)",
                                                 "Images(*.png *.jpg *.jpeg *.bmp)")  # 文件夹路径

        if len(directory1[0]) > 1:
            pix = QPixmap(directory1[0])
            pix = pix.scaled(QSize(200, 200))
            self.lb_old.setPixmap(pix)
            di = QDialog(self)
            di.resize(270, 200)

            label = QLabel(di)
            label.setText("请选择要分割出来的颜色")
            label.move(120,20)

            cb1 = QCheckBox(di)
            cb1.move(20, 50)
            cb1.setText("黑")
            # cb1.setChecked(True)

            cb2 = QCheckBox(di)
            cb2.move(70, 50)
            cb2.setText("灰")
            # cb2.setChecked(True)

            cb3 = QCheckBox(di)
            cb3.move(120, 50)
            cb3.setText("白")
            # cb3.setChecked(True)

            cb4 = QCheckBox(di)
            cb4.move(170, 50)
            cb4.setText("红")
            # cb4.setChecked(True)

            cb5 = QCheckBox(di)
            cb5.move(220, 50)
            cb5.setText("橙")
            # cb5.setChecked(True)

            cb6 = QCheckBox(di)
            cb6.move(20, 70)
            cb6.setText("黄")
            # cb6.setChecked(True)

            cb7 = QCheckBox(di)
            cb7.move(70, 70)
            cb7.setText("绿")
            # cb7.setChecked(True)

            cb8 = QCheckBox(di)
            cb8.move(120, 70)
            cb8.setText("青")
            # cb8.setChecked(True)

            cb9 = QCheckBox(di)
            cb9.move(170, 70)
            cb9.setText("蓝")
            cb9.setChecked(True)

            cb10 = QCheckBox(di)
            cb10.move(220, 70)
            cb10.setText("紫")
            cb10.setChecked(True)

            btn1 = QPushButton(di)
            btn1.setText("segment")
            btn1.move(120, 120)
            btn1.clicked.connect(lambda: di.accept())
            def finish(val):
                if val == 1:
                    state_list.append(cb1.checkState())
                    state_list.append(cb2.checkState())
                    state_list.append(cb3.checkState())
                    state_list.append(cb4.checkState())
                    state_list.append(cb5.checkState())
                    state_list.append(cb6.checkState())
                    state_list.append(cb7.checkState())
                    state_list.append(cb8.checkState())
                    state_list.append(cb9.checkState())
                    state_list.append(cb10.checkState())
                    QMessageBox.information(self,  # 使用infomation信息框
                                            "提示",
                                            "请选择保存文件的路径",
                                            QMessageBox.Yes)
                    fileName_choose = QFileDialog.getSaveFileName(self, "保存文件", "H:/",
                                                                  "png(*.png);;jpg(*.jpg);;jpeg(*.jpeg);;bmp(*.bmp)",
                                                                  "png(*.png)")
                    if len(fileName_choose[0]) < 1:
                        return
                    image_algorithm.color_segmentation(state_list = state_list,
                                            save_path = fileName_choose[0],
                                            image_path = directory1[0])
                    pix = QPixmap(fileName_choose[0])
                    pix = pix.scaled(QSize(200, 200))
                    self.lb_new.setPixmap(pix)
                    self.lb_text.setText("分割成功")
            di.finished.connect(finish)
            di.open()
        else:
            return

    def image_recognition(self):
        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")  # 文件夹路径
        if len(directory1[0]) > 1:
            pix = QPixmap(directory1[0])
            pix = pix.scaled(QSize(200, 200))
            self.lb_old.setPixmap(pix)
            QMessageBox.information(self,  # 使用infomation信息框
                                    "提示",
                                    "请选择保存文件的路径",
                                    QMessageBox.Yes)
            fileName_choose = QFileDialog.getSaveFileName(self, "保存文件", "H:/",
                                                          "png(*.png);;jpg(*.jpg);;jpeg(*.jpeg);;bmp(*.bmp)",
                                                          "png(*.png)")
            if len(fileName_choose[0]) < 1:
                return
            image_algorithm.image_recognition(directory1[0],fileName_choose[0])
            pix = QPixmap(fileName_choose[0])
            pix = pix.scaled(QSize(200, 200))
            self.lb_new.setPixmap(pix)
            self.lb_text.setText("识别成功")
        else:
            return

    def open_map(self):
        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")  # 文件夹路径
        pix = QPixmap(directory1[0])
        pix = pix.scaled(QSize(200, 200))
        self.lb_old.setPixmap(pix)

    def menu(self):
        bar = QMenuBar(self)

        file = bar.addMenu('文件')
        open = file.addAction('打开')
        open.triggered.connect(self.open_map)
        # save = QAction('保存', self)
        # save.setShortcut('Ctrl+S')
        # file.addAction(save)
        # save.triggered.connect(lambda: print("save"))
        quit = file.addAction("退出")
        quit.triggered.connect(lambda: sys.exit())

        run = bar.addMenu('运行')
        accuracy = run.addAction("预测精度")
        accuracy.setShortcut("F5")
        accuracy.triggered.connect(self.action_accuracy)

        image_process = bar.addMenu('图像处理')
        image_classification = image_process.addMenu("图像分类")
        single_classification = image_classification.addAction("单次分类")
        batch_classification = image_classification.addAction("批量分类")

        image_segmentation = image_process.addMenu("图像分割")
        gray_segmentation = image_segmentation.addAction("灰度分割")
        color_segmentation = image_segmentation.addAction("颜色分割")

        image_recognition = image_process.addAction("图像识别")
        single_classification.setShortcut('Ctrl+A')
        batch_classification.setShortcut('Ctrl+B')
        gray_segmentation.setShortcut('Ctrl+W')
        color_segmentation.setShortcut('Ctrl+Q')
        image_recognition.setShortcut('Ctrl+E')
        single_classification.triggered.connect(self.single_classification)
        batch_classification.triggered.connect(self.batch_classification)
        gray_segmentation.triggered.connect(self.gray_segmentation)
        color_segmentation.triggered.connect(self.color_segmentation)
        image_recognition.triggered.connect(self.image_recognition)

    def set_ui(self):
        self.window_init()
        self.map_label()
        self.menu()

    def keyPressEvent(self, evt):
        print(evt.key())
        if evt.key() == Qt.Key_Tab:
            print("Tab")
        elif evt.key() == Qt.Key_S and evt.modifiers() == Qt.ControlModifier | Qt.ShiftModifier:
            print("control + shift + 5")

    def mousePressEvent(self, QMouseEvent):
        print("mouse press")
        if QMouseEvent.button() == Qt.LeftButton:
            self.move_flag = True
            self.mouse_x = QMouseEvent.globalX()
            self.mouse_y = QMouseEvent.globalY()

            self.origin_x = self.x()
            self.origin_y = self.y()

    def mouseReleaseEvent(self, QMouseEvent):
        print("mouse release")
        self.move_flag = False
        self.setWindowOpacity(1)

    def mouseMoveEvent(self, QMouseEvent):
        if self.move_flag == True :
            move_x = QMouseEvent.globalX() - self.mouse_x
            move_y = QMouseEvent.globalY() - self.mouse_y
            print("mouse move")
            self.move(self.origin_x + move_x, self.origin_y + move_y)
            self.setWindowOpacity(0.5)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    window = MyWindow()
    window.show()

    sys.exit(app.exec())

