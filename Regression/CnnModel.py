import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=21, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=19, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=17, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn. Linear(62912,1) #8960 ,17920
        self.drop = nn.Dropout(0.2)

    def forward(self,out):
      out = self.conv1(out)
      out = self.conv2(out)
      out = self.conv3(out)
      out = out.view(out.size(0),-1)
      # print(out.size(1))
      out = self.fc(out)
      return out




class AlexNet(nn.Module):
    def __init__(self, num_classes=1, reduction=16, attention=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
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
            # nn.LeakyReLU(inplace=True),
            # conv1
            # nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm1d(num_features=64),
            # nn.ReLU(inplace=True),
            # # nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # # conv2
            # nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm1d(num_features=32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=5, stride=2),
            # # conv3
            # nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=5, stride=2),
            # # conv4
            # nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=5, stride=2),
            # SELayer(256, reduction),
            # nn.LeakyReLU(inplace=True),

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










class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)

        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out




class DeepSpectra(nn.Module):
    def __init__(self):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
        )
        self.Inception = Inception(16, 32, 32, 32, 96)
        self.fc = nn.Sequential(
            nn.Linear(33120, 5000),
            nn.Dropout(0.5),
            nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

