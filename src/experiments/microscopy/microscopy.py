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

    return model


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
