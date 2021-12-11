import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math

__all__ = ['hcgnet_A1', 'hcgnet_A2']

'''
Yang et al. Gated Convolutional Networks with Hybrid Connectivity for Image Classification. AAAI-2020.
https://github.com/winycg/HCGNet
'''

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1):
        super(BasicConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, groups=groups, bias=False)


    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class _SMG(nn.Module):
    def __init__(self, in_channels, growth_rate,
                 bn_size=4, groups=4, reduction_factor=2, forget_factor=2):
        super(_SMG, self).__init__()
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor
        self.forget_factor = forget_factor
        self.growth_rate = growth_rate
        self.conv1_1x1 = BasicConv(in_channels, bn_size * growth_rate, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                                   padding=1, groups=groups)

        # Mobile
        self.conv_3x3 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=1, groups=growth_rate,)
        self.conv_5x5 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=2, groups=growth_rate, dilation=2)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(growth_rate, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * growth_rate, 2 * growth_rate // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * growth_rate // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(growth_rate // self.forget_factor)
        self.fc1 = nn.Conv2d(growth_rate, growth_rate // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(growth_rate // self.forget_factor, growth_rate, kernel_size=1)

    def forward(self, x):
        x_dense = x
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1).reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))


        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5
        x = x_shortcut * x_shortcut_weight + new_x

        return torch.cat([x_dense, x], 1)


class _HybridBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_HybridBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('SMG%d' % (i+1),
                            _SMG(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, forget_factor=4, reduction_factor=4):
        super(_Transition, self).__init__()
        self.in_channels = in_channels
        self.forget_factor = forget_factor
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.reduce_channels = (in_channels - out_channels) // 2
        self.conv1_1x1 = BasicConv(in_channels, in_channels-self.reduce_channels, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(in_channels-self.reduce_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, groups=1)
        # Mobile
        # Mobile
        self.conv_3x3 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, groups=out_channels)
        self.conv_5x5 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=2, dilation=2, groups=out_channels)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * out_channels, 2 * out_channels // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * out_channels // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(out_channels // self.forget_factor)
        self.fc1 = nn.Conv2d(out_channels, out_channels // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // self.forget_factor, out_channels, kernel_size=1)


    def forward(self, x):
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1)
        forget_context_weight = forget_context_weight.reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))


        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5

        x = x_shortcut * x_shortcut_weight + new_x

        return x

class HCGNet(nn.Module):
    def __init__(self, growth_rate=(8, 16, 32), block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(HCGNet, self).__init__()
        num_init_feature = 2 * growth_rate[0]

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('HybridBlock%d' % (i+1),
                                     _HybridBlock(num_layers, num_feature, bn_size, growth_rate[i]))
            num_feature = num_feature + growth_rate[i] * num_layers
            if i != len(block_config)-1:
                self.features.add_module('Transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.classifier = nn.Linear(num_feature, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = F.adaptive_avg_pool2d(F.relu(features),(1, 1))
        out = features.view(features.size(0), -1)
        embedding = out
        out = self.classifier(out)
        return out, embedding


class HCGNet_n(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=100, number_net=2):
        super(HCGNet_n, self).__init__()
        self.number_net = number_net

        self.module_list = nn.ModuleList([])
        for i in range(number_net):
            self.module_list.append(HCGNet(growth_rate=growth_rate,
                                           block_config=block_config,
                                           num_classes=num_classes))

    def forward(self, x):
        logits = []
        embeddings = []
        for i in range(self.number_net):
            log, emb = self.module_list[i](x)
            logits.append(log)
            embeddings.append(emb)
        return logits, embeddings


class HCGNet_b(nn.Module):
    def __init__(self, growth_rate=(8, 16, 32), block_config=(6,12,24,16), number_net=4,
                 bn_size=4, theta=0.5, num_classes=10):
        super(HCGNet_b, self).__init__()
        num_init_feature = 2 * growth_rate[0]
        self.num_branches = number_net
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            if i == 0:
                self.features.add_module('HybridBlock%d' % (i+1),
                                        _HybridBlock(num_layers, num_feature, bn_size, growth_rate[i]))
                num_feature = num_feature + growth_rate[i] * num_layers

                self.features.add_module('Transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)
            else:
                if i == 1:
                    tmp_num_feature = num_feature
                    for j in range(self.num_branches):
                        setattr(self, 'layer2_HybridBlock' + str(j), _HybridBlock(num_layers, tmp_num_feature, bn_size, growth_rate[i]))
                        num_feature = tmp_num_feature + growth_rate[i] * num_layers
                        setattr(self, 'layer2_Transition' + str(j), _Transition(num_feature,
                                                                         int(num_feature * theta)))
                        num_feature = int(num_feature * theta)

                if i == 2:
                    tmp_num_feature = num_feature
                    for j in range(self.num_branches):
                        setattr(self, 'layer3_HybridBlock' + str(j), _HybridBlock(num_layers, tmp_num_feature, bn_size, growth_rate[i]))
                        num_feature = tmp_num_feature + growth_rate[i] * num_layers
                        setattr(self, 'norm5_' + str(j), nn.BatchNorm2d(num_feature))
                        setattr(self, 'linear_' + str(j), nn.Linear(num_feature, num_classes))
    
    def forward(self, x):
        x = self.features(x)

        logits = []
        input = x
        embedding = []
        for i in range(self.num_branches):
            x = getattr(self, 'layer2_HybridBlock' + str(i))(input)
            x = getattr(self, 'layer2_Transition' + str(i))(x)
            x = getattr(self, 'layer3_HybridBlock' + str(i))(x)
            x = getattr(self, 'norm5_' + str(i))(x)
            x = F.adaptive_avg_pool2d(F.relu(x),(1, 1))
            
            x = x.view(x.size(0), -1)
            embedding.append(x)
            x = getattr(self, 'linear_' + str(i))(x)
            logits.append(x)

        return logits, embedding


def hcgnet_A1(num_classes, number_net):
    if number_net == 2:
        return HCGNet_n(growth_rate=(12, 24, 36), block_config=(8, 8, 8), num_classes=num_classes, number_net=number_net)
    else:
        return HCGNet_b(growth_rate=(12, 24, 36), block_config=(8, 8, 8), num_classes=num_classes, number_net=number_net)

def hcgnet_A2(num_classes, number_net):
    if number_net == 2:
        return HCGNet_n(growth_rate=(24, 36, 64), block_config=(8, 8, 8), num_classes=num_classes, number_net=number_net)
    else:
        return HCGNet_b(growth_rate=(24, 36, 64), block_config=(8, 8, 8), num_classes=num_classes, number_net=number_net)


if __name__ == '__main__':
    net = hcgnet_A1(num_classes=100, number_net=2)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))