import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['wrn_16_2', 'wrn_40_2', 'wrn_28_4']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        embedding = out
        out = self.fc(out)
        return out, embedding

class WideResNet_n(nn.Module):
    def __init__(self, depth=16, widen_factor=2, num_classes=100, number_net=4):
        super(WideResNet_n, self).__init__()
        self.number_net = number_net

        self.module_list = nn.ModuleList([])
        for i in range(number_net):
            self.module_list.append(WideResNet(num_classes=num_classes,
                                           depth=depth, widen_factor=widen_factor))

    def forward(self, x):
        logits = []
        embeddings = []
        for i in range(self.number_net):
            logit, embedding = self.module_list[i](x)
            logits.append(logit)
            embeddings.append(embedding)
        return logits, embeddings


class WideResNet_b(nn.Module):
    def __init__(self, depth, num_classes=100, number_net=4, 
                 widen_factor=1, dropRate=0.0,
                 zero_init_residual=False):
        super(WideResNet_b, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock

        self.number_net = number_net
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        for i in range(self.number_net):

            # 2nd block
            setattr(self, 'block2_' + str(i),
                    NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate))
            # 3rd block
            setattr(self, 'block3_' + str(i),
                    NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate))
            # global average pooling and classifier
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(nChannels[3]))
            setattr(self, 'classifier_' + str(i), nn.Linear(nChannels[3], num_classes))

            self.nChannels = nChannels[3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)

        logits = []
        embedding = []
        input = x
        for i in range(self.number_net):
            x = getattr(self, 'block2_' + str(i))(input)
            x = getattr(self, 'block3_' + str(i))(x)
            x = getattr(self, 'bn_' + str(i))(x)
            x = F.relu(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            embedding.append(x)
            x = getattr(self, 'classifier_' + str(i))(x)
            logits.append(x)

        return logits, embedding

    # Allow for accessing forward method in a inherited class
    forward = _forward


def wrn_16_2(num_classes, number_net):
    if number_net == 2:
        arch = WideResNet_n
    else:
        arch = WideResNet_b
    return arch(depth=16, widen_factor=2, num_classes=num_classes, number_net=number_net)


def wrn_40_2(num_classes, number_net):
    if number_net == 2:
        arch = WideResNet_n
    else:
        arch = WideResNet_b
    return arch(depth=40, widen_factor=2, num_classes=num_classes, number_net=number_net)


def wrn_28_4(num_classes, number_net):
    if number_net == 2:
        arch = WideResNet_n
    else:
        arch = WideResNet_b
    return arch(depth=28, widen_factor=4, num_classes=num_classes, number_net=number_net)



if __name__ == '__main__':
    import torch
    x = torch.randn(1, 3, 32, 32)
    net = wrn_28_4(num_classes=100, number_net=2)
    logits, embedding = net(x)

    from utils import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
