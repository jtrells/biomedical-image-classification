import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict 
from torch import nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class ShallowResNet(pl.LightningDataModule):
    """
    Modification on TorchVision ResNet to allow different configurations per block
    based on params. For example, [2, 2, 2, 1] is a modification of ResNet18 without
    a convblock in block4, and [2, 2, 2, 0] is a shallower version where all the
    layer4 is disregarded.
    """

    def __init__(self, name, num_classes, layers, lr=1e-3):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.layers = layers
        self.lr = lr
        
        self.model = self._get_shallower_model()

    def _get_shallower_model(self):
        if self.name == "resnet18" or self.name == "resnet34":
            model = ShallowTorchResNet(BasicBlock, self.layers)
        else:
            model = ShallowTorchResNet(Bottleneck, self.layers)
        imagenet_dict = load_state_dict_from_url(model_urls[self.name])

        curr_state = model.state_dict()
        new_state = OrderedDict()

        # iterate and get rid of keys not being used
        for key in curr_state.keys():
            if key in imagenet_dict.keys():
                new_state[key] = imagenet_dict[key]

        model.load_state_dict(new_state)
        # there is no much verification but we assume that layers
        # has on index with a value bigger than 0 and the rest can be zeros
        if self.name == 'resnet18' or self.name == 'resnet34':
            if 0 in self.layers:
                zero_idx = self.layers.index(0)
                if zero_idx == 3:
                    num_features = 256
                elif zero_idx == 2:
                    num_features = 128
                elif zero_idx == 1:
                    num_features = 64
            else:
                num_features = 512
        else:
            if 0 in self.layers:
                zero_idx = self.layers.index(0)
                if zero_idx == 3:
                    num_features = 1024
                elif zero_idx == 2:
                    num_features = 512
                elif zero_idx == 1:
                    num_features = 256
            else:
                num_features = 2048
        model.fc = nn.Linear(num_features, self.num_classes) 

        for param in model.parameters():
            param.requires_grad = True
        return model




class ShallowTorchResNet(nn.Module):    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ShallowTorchResNet, self).__init__()
        self.layers = layers
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.layers[0] > 0:
            self.layer1 = self._make_layer(block, 64, layers[0])
        if self.layers[1] > 0:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        if self.layers[2] > 0:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if self.layers[3] > 0:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)                    
                    
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.layers[0] > 0:
            x = self.layer1(x)
        if self.layers[1] > 0:
            x = self.layer2(x)
        if self.layers[2] > 0:
            x = self.layer3(x)
        if self.layers[3] > 0:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
