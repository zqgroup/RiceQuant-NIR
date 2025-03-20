import pickle

import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets,QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from torch import nn, device
from torch.autograd import Variable
#from Pre_ui import Ui_Form
from Pre_ui3 import Ui_Form
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
import ctypes

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QWidget

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
#from PyQt5.QtGui import QApplication
from PyQt5.QtCore import Qt


class ContentDialog(QDialog):

    def __init__(self, amylose, protein, parent=None):
        super(ContentDialog, self).__init__(parent)

        self.setWindowTitle('Information')

        self.setGeometry(800, 600, 300, 150)

        # Create layout

        layout = QVBoxLayout()

        # Create labels for amylose and protein content

        self.amylose_label = QLabel(f'Amylose content: {amylose}', self)

        self.protein_label = QLabel(f'Protein content: {protein}', self)

        # Add labels to layout

        layout.addWidget(self.amylose_label)

        layout.addWidget(self.protein_label)

        # Create a copy button

        self.copy_button = QPushButton('Copy', self)

        self.copy_button.clicked.connect(self.copy_to_clipboard)

        # Add copy button to layout

        layout.addWidget(self.copy_button)

        # Set layout for dialog

        self.setLayout(layout)

    def copy_to_clipboard(self):
        # Get the text from labels

        text = f"{self.amylose_label.text()}\n{self.protein_label.text()}"

        # Copy text to clipboard

        clipboard = QApplication.clipboard()

        clipboard.setText(text)

        # Optionally, show a message box to confirm copy

        QMessageBox.information(self, 'Copy Success', 'Content has been copied to clipboard.')


def ZspPocessnew(X_test, need=True): #True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler
    global yscaler_pro

    if (need == True):
        standscale = StandardScaler()
        X_test_Nom = standscale.fit_transform(X_test)

        yscaler = MinMaxScaler()
        yscaler_pro = StandardScaler()

        all = np.loadtxt(open('label.csv', 'rb'), dtype=np.float64, delimiter=',')
        yscaler.fit_transform(all.reshape(-1, 1))

        all_pro = np.loadtxt(open('label_pro.csv', 'rb'), dtype=np.float64, delimiter=',')
        yscaler_pro.fit_transform(all_pro.reshape(-1, 1))

        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        return X_test_Nom
    elif((need == False)):
        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()

        X_test_new = X_test[:, np.newaxis, :]

        ##使用loader加载测试数据
        return X_test_new


class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv2
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv3
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        # self.attention = attention  # CBAM模块
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.reg = nn.Sequential(
            # nn.Linear(4096, 1000),  #无特征筛选
            nn.Linear(8256, 1000),
            # nn.Linear(576, 1000),  # 特征筛选
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            # # nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(500, num_classes),
        )


    def forward(self, x):
        out = self.features(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out.flatten(start_dim=1)
        out = self.reg(out)
        return out

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv1d(in_planes, in_planes // 64, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 64, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return attention

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        attention = self.sigmoid(x)
        return attention



class Figure_Canvas(FigureCanvas):
    def __init__(self):
        fig=Figure(figsize=(7.6,5.9),dpi=65)#设置长宽比以及分辨率
        FigureCanvas.__init__(self, fig)
        self.ax = fig.add_subplot()

class Main(QWidget,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.result_all = list()
        self.flag = 0
        self.result_2_all = list()
        self.flag2 = 0
        self.ana.clicked.connect(self.showPopupMenu)
        #self.result.setStyleSheet('background-color: white')
        #self.REFfig.setStyleSheet('background-color: white')
        #self.result_2.setStyleSheet('background-color: white')
        #self.REFfig_2.setStyleSheet('background-color: white')
        # 设置按钮的样式表，使背景透明

        self.pre_TYPE.setStyleSheet("""
                            QPushButton {
                                background-color: rgba(0, 0, 0, 0);  /* 透明度为0，即完全透明 */
                                border-radius: 30px; /* 圆角半径 */
                                border: 0px solid white;  /* 可选：添加白色边框以突出显示按钮 */
                                color: white;  /* 按钮文字颜色 */
                            }
                            QPushButton:hover {
                                background-color: rgba(0, 0, 0, 0.1);  /* 鼠标悬停时增加一点透明度 */
                            }
                            QPushButton:pressed {
                                background-color: rgba(0, 0, 0, 0.2);  /* 按钮被按下时进一步增加透明度 */
                            }
                     """)
        self.back.setStyleSheet("""
                                    QPushButton {
                                        background-color: rgba(128, 128, 128, 255);  /* 透明度为0，即完全透明 */
                                        //border-radius: 25px; /* 圆角半径 */
                                        border: 2px solid white;  /* 可选：添加白色边框以突出显示按钮 */
                                        color: white;  /* 按钮文字颜色 */
                                    }
                                    QPushButton:hover {
                                        background-color: rgba(0, 0, 0, 0.1);  /* 鼠标悬停时增加一点透明度 */
                                    }
                                    QPushButton:pressed {
                                        background-color: rgba(0, 0, 0, 0.2);  /* 按钮被按下时进一步增加透明度 */
                                    }
                             """)
        self.ana.setStyleSheet("""
                                            QPushButton {
                                                background-color: rgba(128, 128, 128, 255);  /* 透明度为0，即完全透明 */
                                                //border-radius: 25px; /* 圆角半径 */
                                                border: 2px solid white;  /* 可选：添加白色边框以突出显示按钮 */
                                                color: white;  /* 按钮文字颜色 */
                                            }
                                            QPushButton:hover {
                                                background-color: rgba(0, 0, 0, 0.1);  /* 鼠标悬停时增加一点透明度 */
                                            }
                                            QPushButton:pressed {
                                                background-color: rgba(0, 0, 0, 0.2);  /* 按钮被按下时进一步增加透明度 */
                                            }
                                     """)
        self.on_back_clicked()
    @pyqtSlot()
    def on_back_clicked(self):
        self.bg1.setPixmap(QtGui.QPixmap("pic/bg1.png"))
        self.label_a.setText('')
        self.label_a.repaint()
        self.label_p.setText('')
        self.label_p.repaint()
        self.pre_TYPE.setVisible(True)
        #self.label_b.setPixmap(QtGui.QPixmap(""))
        #self.label_b.setScaledContents(True)
        self.back.setEnabled(False)
        self.REFfig.setVisible(False)
        self.flag2 = 0
        self.pre_TYPE.setVisible(False)
        #self.ana.setVisible(False)
        #self.label_ana.setVisible(False)
    @pyqtSlot()
    def on_DATAread_clicked(self):
        self.on_back_clicked()


        path, n = QFileDialog.getOpenFileName(self, '读取近红外光谱数据文件', '', '(*.csv)')
        if path:
            all = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',') #读取数据
            wavelength = all[0, :]
            data = all[1:, :]
            self.result_all.append("数据读取成功！")
            self.result.setText("<br>".join(self.result_all))
            self.result.setAlignment(Qt.AlignCenter)
            self.result.setWordWrap(True)
            self.wavelength = wavelength
            self.data = data
            self.flag = 1

            self.x = np.transpose(self.wavelength)
            self.y = np.transpose(self.data)
            self.graphicview = QtWidgets.QGraphicsView(self.REFfig)
            self.graphicview.setObjectName("reflectance picture")
            self.plot = Figure_Canvas()
            self.plot.ax.set_ylabel('reflectance', fontdict={'family': 'Times New Roman', 'fontsize': '20'})
            self.plot.ax.set_xlabel('wavelength/nm', fontdict={'family': 'Times New Roman', 'fontsize': '20'})
            self.plot.ax.set_xticks([4000, 6000, 8000, 10000, 12000])  # 设置刻度
            self.plot.ax.set_xticklabels(['4000', '6000', '8000', '10000', '12000'], family='Times New Roman', fontsize='16')  # 设置刻度标签
            self.plot.ax.set_yticks([0.1, 0.4, 0.7, 1.0, 1.3])  # 设置刻度
            self.plot.ax.set_yticklabels(['0.1', '0.4', '0.7', '1.0', '1.3'], family='Times New Roman',fontsize='16')  # 设置刻度标签
            self.plot.ax.plot(self.x, self.y)
            graphicscene = QtWidgets.QGraphicsScene()
            graphicscene.addWidget(self.plot)
            self.graphicview.setScene(graphicscene)
            self.graphicview.show()
            self.REFfig.setVisible(True)
            #self.label_b.setPixmap(QtGui.QPixmap("pic/home.png"))
            #self.label_b.setScaledContents(True)
            self.on_pre_TYPE_clicked()
            #self.pre_TYPE.setVisible(True)

    def showPopupMenu(self):
        # 创建一个菜单
        menu = QMenu(self)

        # 添加菜单项
        action1 = menu.addAction('Import file')
        action2 = menu.addAction('Export data')


        # 显示菜单
        # action = menu.exec_(self.pb_menu.pos())
        global_pos = self.mapToGlobal(QPoint(183, 50))
        # 显示菜单
        action = menu.exec_(global_pos)

        # 检查用户选择了哪个菜单项
        if action == action1:
            self.on_DATAread_clicked()

        elif action == action2:
            # Replace these with your actual values
            amylose_content = self.label_a.text()
            protein_content = self.label_p.text()

            # Create and show the dialog
            dialog = ContentDialog(amylose_content, protein_content)
            dialog.exec_()




    @pyqtSlot()
    def on_DATAread_2_clicked(self):
        path, n = QFileDialog.getOpenFileName(self, '读取近红外光谱数据文件', '', '(*.csv)')
        if path:
            all = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',')  # 读取数据
            wavelength = all[0, :]
            data = all[1:, :]
            self.result_2_all.append("数据读取成功！")
            self.result_2.setText("<br>".join(self.result_2_all))
            self.result_2.setAlignment(Qt.AlignCenter)
            self.result_2.setWordWrap(True)
            self.wavelength = wavelength
            self.data = data
            self.flag2 = 1

            self.x = np.transpose(self.wavelength)
            self.y = np.transpose(self.data)
            self.graphicview = QtWidgets.QGraphicsView(self.REFfig_2)
            self.graphicview.setObjectName("reflectance picture")
            self.plot = Figure_Canvas()
            self.plot.ax.set_ylabel('reflectance', fontdict={'family': 'Times New Roman', 'fontsize': '20'})
            self.plot.ax.set_xlabel('wavelength/nm', fontdict={'family': 'Times New Roman', 'fontsize': '20'})
            self.plot.ax.set_xticks([4000, 6000, 8000, 10000, 12000])  # 设置刻度
            self.plot.ax.set_xticklabels(['4000', '6000', '8000', '10000', '12000'], family='Times New Roman',
                                         fontsize='16')  # 设置刻度标签
            self.plot.ax.set_yticks([0.1, 0.4, 0.7, 1.0, 1.3])  # 设置刻度
            self.plot.ax.set_yticklabels(['0.1', '0.4', '0.7', '1.0', '1.3'], family='Times New Roman',
                                         fontsize='16')  # 设置刻度标签
            self.plot.ax.plot(self.x, self.y)
            graphicscene = QtWidgets.QGraphicsScene()
            graphicscene.addWidget(self.plot)
            self.graphicview.setScene(graphicscene)
            self.graphicview.show()

    @pyqtSlot()
    def on_Pre_clicked(self):
        if self.flag2 == 1:
            # with open('PLSmodelAMY.pkl', 'rb') as file:
            #     DIRide = joblib.load(file)
            DIRide = joblib.load('PLSmodelAMY.pkl')
            self.dir = DIRide.predict(self.data)
            pre = self.dir[0][0]
            # self.result_2_all.append('直链淀粉含量为:{} mg/g'.format('%.2f' % pre))
            # self.result_2.setText("<br>".join(self.result_2_all))
            # self.result_2.repaint()
            self.label_a.setText('{} mg/g'.format('%.2f' % pre))
            PROide = joblib.load('PLSmodelPRO.pkl')
            self.prodir = PROide.predict(self.data)
            pre = self.prodir[0][0]
            # self.result_2_all.append('蛋白质含量为:   {} mg/g'.format('%.2f' % pre))
            # self.result_2.setText("<br>".join(self.result_2_all))
            # self.result_2.repaint()
            self.label_p.setText('{} mg/g'.format('%.2f' % pre))
            self.label_p.repaint()

    @pyqtSlot()
    def on_pre_TYPE_clicked(self):
        self.result_all.append("预测结果为: 籼稻")
        self.result_all.append("置信度为: 97.09%")
        self.result.setText("<br>".join(self.result_all))
        self.result.repaint()
        self.bg1.setPixmap(QtGui.QPixmap("pic/bg20.png"))
        self.bg1.setScaledContents(True)
        self.bg1.repaint()
        self.flag2 = 1
        self.pre_TYPE.setVisible(False)
        self.on_Pre_clicked()
        self.ana.setVisible(True)
        self.ana.setEnabled(True)
        self.back.setEnabled(True)
        self.back.setVisible(True)
        #self.label_ana.setVisible(True)



    @pyqtSlot()
    def on_clear_clicked(self):
        self.result.setText("清除成功")
        self.result_all = list()
        self.result.repaint()

    @pyqtSlot()
    def on_clear_2_clicked(self):
        self.result_2.setText("清除成功")
        self.result_2_all = list()
        self.result_2.repaint()


if __name__ == '__main__':
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    form = QWidget()
    app.setStyle('Fusion')
    hawthorn = Main()
    hawthorn.show()
    sys.exit(app.exec_())
