from numpy import imag
import torch
from torch import Tensor, nn
from .res50_backbone import resnet50


class Backbone(nn.Module):

    def __init__(self, pretrain_path=None):
        '''
        input:
            pretrain_path: 预训练模型
        '''
        super(Backbone, self).__init__()
        net = resnet50()

        # feature_map 1-6 的输出通道数分别是: 
        # 1024, 512, 512, 256, 256, 256
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None: # 加载预训练模型
            net.load_state_dict(torch.path(pretrain_path))

        # ResNet50中: conv1, bn1, relu, maxpool, layer1, layer2, layer3
        self.feature_extractor = nn.Sequential(*list(net.children())[: 7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 将conv4_block1的stride修改为1
        conv4_block1.conv1.stride = (1, 1) # ResNet50中可省略
        conv4_block1.conv2.stride = (1, 1) # 3x3 卷积中的 stride
        conv4_block1.downsample[0].stride = (1, 1) # shortcut 中的 stride

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

        # 生成 additional_layer 1-5
        self._build_additional_features(self.feature_extractor.out_channels)

        # feature_map 1-6 中每个 cell 生成的 default_box 数量
        self.num_defaults = [4, 6, 6, 6, 4, 4]

        # 位置提取器
        location_extractors = []

        # 类别提取器
        confidence_extractors = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd, number_default_boxes
            # oc, output_channel
            location_extractors.append(
                nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1)
            )
            confidence_extractors.append(
                nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1)
            )

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleDict(confidence_extractors)
        self._init_weights()

        pass

    def _build_additional_features(self, input_size):
        '''
        describe:
            为backbone添加额外的一系列卷积层, 得到相应的一系列特征提取器
            即 additional_layer
        params:
            input_size: feature map 的 channels
        '''
        additional_blocks = []

        # feature_map 2-6 中第1个卷积层的 channels, 即
        # additional_layer 1-5中第1个卷积层的 channels
        middle_channels = [256, 256, 128, 128, 128]

        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            # input_ch 分别为: 1024, 512, 512, 256, 256
            # output_ch 分别为: 512, 512, 256, 256, 256
            # middle_ch 分别为: 256, 256, 128, 128, 128
            
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True)
            )
            additional_blocks.append(layer)

        self.addition_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        '''
        describe:
            模型参数初始化
        '''
        layers = [*self.addition_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        '''
        params:
        '''
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch_size, n * 4, feat_size, feat_size] --> [batch_size, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))

            # [batch_size, n * classes, feat_size, feat_size] --> [batch_size, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        # contiguous() 方法: 连续存储
        locs, confs = torch.cat(locs, dim=2).contiguous(), torch.cat(confs, dim=2).contiguous()

        return locs, confs

    def forward(self, image, targets=None):
        # x 是 Conv4 的输出
        x = self.feature_extractor(image)

        # 存储每一个 feature_map 的输出
        # feature_map 1-6的输出 shape 分别是:
        # 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.addition_blocks:
            x = layer(x)
            detection_features.append(x)

        locs, confs = self.bbox_view()

        pass