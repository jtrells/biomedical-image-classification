import torch
import torch.nn.functional as F
import torch.nn as nn


# Class to Extract Feature of the previous layer to softmax in Resnet
class ResNetFeatureExtractor(nn.Module):
            def __init__(self,model):
                super(ResNetFeatureExtractor, self).__init__()
                self.features = nn.Sequential(
                    *list(model.children())[:-1]
                )
                
            def forward(self, x):
                x = self.features(x)
                return x


