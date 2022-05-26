from tkinter.messagebox import NO
import torch
from torch import nn 


class Bottleneck(nn.Module):
    '''Residual'''

    # ResNet50中, 每个Residual的输出channel是输入channel的4倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        '''
        params:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 步幅
            downsample: shortcut中是否使用1x1卷积
        '''
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel * self.expansion, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        '''
        params:
            block: Bottlenect
            blocks_num: 每个block中residual的个数, ResNet50中, conv2-conv5分别是[3,4,6,3]
            num_classes: 分类类别数, ImageNet中是1000类
            include_top: 
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # ResNet中conv1的输入channel数
        
        # Conv1
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True) # inplace???
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        # Conv3
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        # Conv4
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        '''
        input:
            block: Residual
            channel: 输入通道数
            block_num: Residual的个数
            stride: 步幅
        '''
        downsample = None

        # Conv2-Conv5的第一个Residual的shortcut使用1x1卷积, 将输入channel数扩大4倍
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, include_top=include_top)


# net = resnet50()
# print(net)