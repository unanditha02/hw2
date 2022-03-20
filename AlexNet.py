import numpy as np

import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import torchvision.models as models

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# LocalizerAlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace)
#     (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace)
#     (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace)
#   )
#   (classifier): Sequential(
#     (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
#   )
# )

class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,stride=4,padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3,stride=2,dilation=1,ceil_mode=False),
                        nn.Conv2d(64, 192, kernel_size=5,stride=1,padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3,stride=2,dilation=1,ceil_mode=False),
                        nn.Conv2d(192,384, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(384, 256, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(inplace=True)) 

        self.classifier = nn.Sequential(nn.Conv2d(256,256, kernel_size=3,stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256, kernel_size=1,stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256,num_classes, kernel_size=1,stride=1))

    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model


    def forward(self, x):
        #TODO: Define fwd pass


        return x


def localizer_alexnet(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not
    model_dict = model.state_dict()

    if pretrained:
        alexnet_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
        for name, param in alexnet_dict.items():
            if 'features' in name:
                model_dict[name] = param
            if 'classifier' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'classifier' in name and 'bias' in name:
                nn.init.zeros_(param)
            print(name)
    else:
        for name, param in model_dict.items():
            # if 'features' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.zeros_(param)
            print(name)


    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed
    

    return model