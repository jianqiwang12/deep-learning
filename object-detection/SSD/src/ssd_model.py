import torch
from torch import nn
from .res50_backbone import resnet50


class Backbone(nn.Module):

    def __init__(self, pretrain_path=None):
        '''
        input:
            pretrain_path: 预训练模型
        '''
        super(Backbone, self).__init__()
        net = resnet50()

        # feature map 1--6 的输出通道书分别是: 1024, 512, 512, 256, 256, 256
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None: # 加载预训练模型
            net.load_state_dict(torch.path(pretrain_path))

        # ResNet50中: conv1, bn1, relu, maxpool, layer1, layer2, layer3
        self.feature_extractor = nn.Sequential(*list(net.children())[: 7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 将conv4_block1的stride修改为1
        conv4_block1.conv1.stride = (1, 1) # ResNet50中可省略
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        
        return x


class SSD300(nn.Module):

    def __init__(self, backbone=None, num_classes=21):
        '''
        params:
            backbone:
            num_classes: 20类+背景
        '''
        super(SSD300, self).__init__()

        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone does not have attribute: out_channels")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = [] # 位置提取器
        confidence_extractors = [] # 类别提取器

    def _build_additional_features(self, input_size):
        '''
        describe:
            为backbone添加额外的一系列卷积层, 得到相应的一系列特征提取器
        params:
            input_size:
        '''
        pass

    def _init_weights(self):
        '''
        describe:
            模型参数初始化
        '''
        pass