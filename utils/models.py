import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)

"""
We provide the models, which might be used in the experiments on FedD3, as follows:
    - AlexNet model customized for CIFAR-10 (AlexCifarNet) with 1756426 parameters
    - LeNet model customized for MNIST with 61706 parameters
    - Further ResNet models
    - Further Vgg models
"""


# AlexNet model customized for CIFAR-10 with 1756426 parameters
class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 4096)
        out = self.classifier(out)
        return out


# LeNet model customized for MNIST with 61706 parameters
class LeNet(nn.Module):
    supported_dims = {28}

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=False)  # 6 x 28 x 28
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14
        out = F.relu(self.conv2(out), inplace=False)  # 16 x 7 x 7
        out = F.max_pool2d(out, 2)   # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5
        out = F.relu(self.fc1(out), inplace=False)
        out = F.relu(self.fc2(out), inplace=False)
        out = self.fc3(out)

        return out

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 → 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 → 7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Further ResNet for ImageNet
# def generate_resnet(num_classes=200, in_channels=3, model_name="ResNet18"):
#     # 1. 动态加载模型和预训练权重
#     weights_dict = {
#         "ResNet18": ResNet18_Weights.DEFAULT,
#         "ResNet34": ResNet34_Weights.DEFAULT,
#         "ResNet50": ResNet50_Weights.DEFAULT,
#         "ResNet101": ResNet101_Weights.DEFAULT,
#         "ResNet152": ResNet152_Weights.DEFAULT
#     }
#
#     if model_name not in weights_dict:
#         raise ValueError(f"Unsupported model_name: {model_name}")
#
#     model_func = getattr(models, model_name.lower())
#     model = model_func(weights=weights_dict[model_name])
#
#     # 2. 折中版核心改造 (针对 Tiny ImageNet 64x64)
#     # 采用 3x3 卷积，stride=2 进行一次温和的降采样 (64x64 -> 32x32)
#     model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
#
#     # 移除 maxpool，防止发生破坏性的二次降采样
#     model.maxpool = nn.Identity()
#
#     # 3. 替换分类头
#     fc_features = model.fc.in_features
#     model.fc = nn.Linear(fc_features, num_classes)
#
#     return model

# LC25000
def generate_resnet(num_classes=5,
                    in_channels=3,
                    model_name="ResNet18"):

    weights = {
        "ResNet18": ResNet18_Weights.DEFAULT,
        "ResNet34": ResNet34_Weights.DEFAULT,
        "ResNet50": ResNet50_Weights.DEFAULT,
    }

    model = getattr(models, model_name.lower())(
        weights=weights[model_name]
    )

    # LC25000 适配
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # 保留 maxpool（速度更平衡）

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model

def generate_resnet_ham10000(
        num_classes=7,
        model_name="ResNet18"
):

    weights_dict = {
        "ResNet18": None,
        "ResNet34": ResNet34_Weights.DEFAULT,
        "ResNet50": ResNet50_Weights.DEFAULT,
        "ResNet101": ResNet101_Weights.DEFAULT,
        "ResNet152": ResNet152_Weights.DEFAULT
    }

    if model_name not in weights_dict:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_func = getattr(models, model_name.lower())

    # ===== 使用 ImageNet 预训练 =====
    model = model_func(weights=weights_dict[model_name])

    # ===== 不修改 conv1 =====
    # HAM10000 非常适合原生 ResNet

    # ===== 替换分类头 =====
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model

# Further Vgg models
def generate_vgg(num_classes=5, in_channels=1, model_name="vgg11"):
    if model_name == "VGG11":
        model = models.vgg11(weights=None)
    elif model_name == "VGG11_bn":
        model = models.vgg11_bn(weights=True)
    elif model_name == "VGG13":
        model = models.vgg11(weights=False)
    elif model_name == "VGG13_bn":
        model = models.vgg11_bn(weights=True)
    elif model_name == "VGG16":
        model = models.vgg11(weights=False)
    elif model_name == "VGG16_bn":
        model = models.vgg11_bn(weights=True)
    elif model_name == "VGG19":
        model = models.vgg11(weights=False)
    elif model_name == "VGG19_bn":
        model = models.vgg11_bn(weights=True)

    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CNN, self).__init__()

        self.fp_con1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        self.ternary_con2 = nn.Sequential(OrderedDict([
            # Conv Layer block 1
            ('conv1', nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # Conv Layer block 2
            ('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # Conv Layer block 3
            ('conv4', nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('norm5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),
        ]))

        # 用 dummy 数据计算全连接层输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 32, 32)
            out = self.fp_con1(dummy_input)
            out = self.ternary_con2(out)
            fc_input_dim = out.view(1, -1).size(1)

        self.fp_fc = nn.Linear(fc_input_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == "__main__":
    model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    for model_name in model_name_list:
        model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)

