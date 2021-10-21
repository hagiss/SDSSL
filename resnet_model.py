import torchvision.models as models
import torch


class ResNet(torch.nn.Module):
    def __init__(self, name='resnet18'):
        super(ResNet, self).__init__()
        if name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            resnet = models.resnet18(pretrained=False)

        self.layer1 = torch.nn.Sequential(*list(resnet.children())[:-5])
        self.layer2 = torch.nn.Sequential(*list(resnet.children())[-5])
        self.layer3 = torch.nn.Sequential(*list(resnet.children())[-4])
        self.layer4 = torch.nn.Sequential(*list(resnet.children())[-3])
        self.avg_pool = torch.nn.Sequential(list(resnet.children())[-2])

        self.embed_dim = resnet.fc.in_features

        # self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        # self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def get_intermediate_layers(self, x):
        output = []
        x = self.layer1(x)
        # print('1', torch.flatten(self.avg_pool(x), 1).shape) 64
        output.append(torch.flatten(self.avg_pool(x), 1))
        x = self.layer2(x)
        # print('2', torch.flatten(self.avg_pool(x), 1).shape) 128
        output.append(torch.flatten(self.avg_pool(x), 1))
        x = self.layer3(x)
        # print('3', torch.flatten(self.avg_pool(x), 1).shape) 256
        output.append(torch.flatten(self.avg_pool(x), 1))
        x = self.layer4(x)
        # print('4', torch.flatten(self.avg_pool(x), 1).shape) 512
        output.append(torch.flatten(self.avg_pool(x), 1))

        return output

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avg_pool(x), 1)
