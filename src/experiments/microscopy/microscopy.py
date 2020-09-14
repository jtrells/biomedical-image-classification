from torchvision import models
from torch import nn
import torch

from collections import OrderedDict 
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet

def get_model(model_name, variant, num_classes, layers=None, pretrained=True):
    if variant == "fine-tuning":
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            model = models.resnet152(pretrained=pretrained)
        # retarget number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif variant == "shallow":
        model = get_shallower_model(model_name, layers, num_classes)
    elif variant == "efficient-net":
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        for param in model.parameters():
             param.requires_grad = True
#         model._fc.weight.requires_grad = True
#         model._fc.bias.requires_grad = True

    return model


def experiment(experiment_name, model_name, num_classes=4, pretrained=True):
    model = get_model(model_name, "fine-tuning", num_classes, pretrained=pretrained)
    if experiment_name == 'fc':
        model = experiment1(model)
    elif experiment_name == 'whole':
        model = experiment2(model)
    elif 'layer4' in experiment_name:
        name, i = experiment_name.split('-')
        model = experiment3(model, int(i))
    elif 'layer3' in experiment_name:
        name, i = experiment_name.split('-')
        model = experiment4(model, int(i))
    elif 'layer2' in experiment_name:
        name, i = experiment_name.split('-')
        model = experiment5(model, int(i))
    elif 'layer1' in experiment_name:
        name, i = experiment_name.split('-')
        model = experiment6(model, int(i))

    return model

# There should be a smarter way to set requires_grad but as I don't know how 
# to access the layers as a dictionary, it's quite tricky. This works for now.
def experiment1(model):
    """
    Experiment 1: Unfreeze only the last fc layer
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def experiment2(model):
    """
    Experiment 2: Fine-tune the whole layers
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def experiment3(model, i):
    """
    Experiment 3: Fine-tune from the last residual block (layer 4)
    """
    if len(model.layer4) <= i:
        raise Exception("Block does not have {0} sublayers".format(i-1))
    
    for param in model.parameters():
        param.requires_grad = False
    
    layers = model.layer4
    for idx in range(len(layers) - i):
        for p in layers[-(idx+1)].parameters():
            p.requires_grad = True
#     for param in model.layer4.parameters():
#         param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def experiment4(model, i):
    """
    Experiment 4: Fine-tune from the third residual block (layer 3)
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    layers = model.layer3
    for idx in range(len(layers) - i):
        for p in layers[-(idx+1)].parameters():
            p.requires_grad = True
#     for param in model.layer3.parameters():
#         param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def experiment5(model, i):
    """
    Experiment 5: Fine-tune from the second residual block (layer 2)
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    layers = model.layer2
    for idx in range(len(layers) - i):
        for p in layers[-(idx+1)].parameters():
            p.requires_grad = True
#     for param in model.layer2.parameters():
#         param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def experiment6(model, i):
    """
    Experiment 6: Fine-tune from the first residual block (layer 1)
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer2.parameters():
        param.requires_grad = True
    layers = model.layer1
    for idx in range(len(layers) - i):
        for p in layers[-(idx+1)].parameters():
            p.requires_grad = True
#     for param in model.layer1.parameters():
#         param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def get_shallower_model(model_name, layers, num_classes):
    if model_name == 'resnet18' or model_name == 'resnet34':
        resnet_ = ShallowResNet(BasicBlock, layers)
    else:
        resnet_ = ShallowResNet(Bottleneck, layers)
    imagenet_dict = load_state_dict_from_url(model_urls[model_name])
    
    curr_state = resnet_.state_dict()
    new_state = OrderedDict()

    # iterate and get rid of keys not being used
    for key in curr_state.keys():
        if key in imagenet_dict.keys():
            new_state[key] = imagenet_dict[key]
        
    resnet_.load_state_dict(new_state)
    # there is no much verification but we assume that layers
    # has on index with a value bigger than 0 and the rest can be zeros
    if model_name == 'resnet18' or model_name == 'resnet34':
        if 0 in layers:
            zero_idx = layers.index(0)
            if zero_idx == 3:
                num_features = 256
            elif zero_idx == 2:
                num_features = 128
            elif zero_idx == 1:
                num_features = 64
        else:
            num_features = 512
    else:
        if 0 in layers:
            zero_idx = layers.index(0)
            if zero_idx == 3:
                num_features = 1024
            elif zero_idx == 2:
                num_features = 512
            elif zero_idx == 1:
                num_features = 256
        else:
            num_features = 2048
    resnet_.fc = nn.Linear(num_features, num_classes)
    
    for param in resnet_.parameters():
        param.requires_grad = True
    
    return resnet_


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
    

class ShallowResNet(nn.Module):
    """
    Same as torchvision ResNet but also allow to incluide or not the layer blocks
    based on the layers param. For example, [2, 2, 2, 1] is a modification of 
    ResNet18 without a conv block in layer 4, and [2, 2, 2, 0] is a shallower version
    where all the layer4 is disregarded.
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ShallowResNet, self).__init__()
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


