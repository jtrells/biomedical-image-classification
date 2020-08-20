from torchvision import models
from torch import nn


def get_model(model_name, pretrained, num_classes):
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

    return model


def experiment(experiment_name, model_name, num_classes=4, pretrained=True):
    model = get_model(model_name, pretrained, num_classes)
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