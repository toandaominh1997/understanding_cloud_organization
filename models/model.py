import torch.nn as nn
import torchvision
import torch 
from modules import Unet


class deeplapModel(nn.Module):
    def __init__(self, num_class=4):
        super(Model, self).__init__()
        self.models = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_class, aux_loss=None)
        # self.models = Unet('efficientnet-b1', classes=num_class, activation='softmax')
    
    def forward(self, inputs):
        out = self.models(inputs)['out']
        return out

class fcnModel(nn.Module):
    def __init__(self, num_class=4):
        super(fcnModel, self).__init__()
        self.models = torchvision.models.segmentation.fcn_resnet50(pretrained=True, num_classes=num_class)
    def forward(self, inputs):
        out = self.models(inputs)
        return out

# if.segmentation.fcn_resnet50__name__=='__main__':
#     model = Model(num_class=4)
#     inputs = torch.randn(1, 3, 600, 600)
#     print(model(inputs).size())
