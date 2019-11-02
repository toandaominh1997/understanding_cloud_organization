import torch.nn as nn
import torchvision
import torch 

class Model(nn.Module):
    def __init__(self, num_class=4):
        super(Model, self).__init__()
        self.models = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_class, aux_loss=None)
    
    def forward(self, inputs):
        out = self.models(inputs)['out']
        return out

if __name__=='__main__':
    model = Model(num_class=4)
    inputs = torch.randn(1, 3, 224, 224)
    print(model(inputs).size())
