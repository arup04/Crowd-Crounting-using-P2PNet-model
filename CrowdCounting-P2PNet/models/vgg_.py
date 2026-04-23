
import torch.nn as nn
import torch
import os
__all__ = ['vgg16_bn']
model_urls = {'vgg16_bn': 'vgg16_bn-6c64b313.pth'}
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features; self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights: self._initialize_weights()
    def forward(self, x):
        x = self.features(x); x = self.avgpool(x); x = torch.flatten(x, 1); x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M': layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else: layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
def vgg16_bn(pretrained=False, progress=True, sync=False, **kwargs):
    if pretrained: kwargs['init_weights'] = False
    model = VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], batch_norm=True), **kwargs)
    if pretrained:
        cwd = os.getcwd()
        w_path = os.path.join(cwd, model_urls['vgg16_bn'])
        if os.path.exists(w_path): model.load_state_dict(torch.load(w_path, map_location='cpu'))
    return model
