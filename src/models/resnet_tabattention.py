import torch.nn as nn
import torch
from tabattention import TabAttention


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, n_frames_last_layer=2,
                 input_last_layer=(7, 7), tabattention=True, n_tab=6, i=1):

        self.tabattention = tabattention

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        if not tabattention:
            self.conv2 = nn.Sequential(
                conv_builder(planes, planes, midplanes),
                nn.BatchNorm3d(planes)
            )
        else:
            f_channels = (n_frames_last_layer * (2 ** (4 - i)))
            input_size = (input_last_layer[0] * (2 ** (4 - i)), input_last_layer[0] * (2 ** (4 - i)))
            self.cbam_layer = TabAttention(input_dim=(planes, input_size[0], input_size[1], f_channels), tab_dim=n_tab)
            self.conv2 = nn.Sequential(
                conv_builder(planes, planes, midplanes),
                nn.BatchNorm3d(planes)
            )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, tab):
        residual = x

        out = self.conv1(x)
        if self.tabattention:
            out = torch.permute(out, (0, 1, 3, 4, 2))
            out = self.cbam_layer(out, tab)
            out = torch.permute(out, (0, 1, 4, 2, 3))

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class DoubleInputSequential(nn.Module):
    def __init__(self, *layers):
        super(DoubleInputSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, tab):
        for l in self.layers:
            x = l(x, tab)
        return x


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, stem, num_classes=1, zero_init_residual=False, head=False,
                 n_frames_last_layer=2, input_last_layer=(7, 7), n_tab=6, ):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.head = head
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1,
                                       n_frames_last_layer=n_frames_last_layer,
                                       input_last_layer=input_last_layer, n_tab=n_tab, i=1)

        if not head:
            self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2,
                                           n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=2)
            self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2,
                                           n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=3)
            self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2,
                                           n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=4)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc_numerical = nn.Linear(6, 1)
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, num_classes)
            )

        # init weights
        self._initialize_weights()

    def forward(self, x, tab=None):
        x = self.stem(x)

        x = self.layer1(x, tab)
        if self.head:
            return x

        x = self.layer2(x, tab)

        x = self.layer3(x, tab)

        x = self.layer4(x, tab)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)

        x = self.fc(x)
        x = x.flatten()
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, n_frames_last_layer=2,
                    input_last_layer=(7, 7), n_tab=6, i=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, conv_builder, stride, downsample,
                  n_frames_last_layer=n_frames_last_layer,
                  input_last_layer=input_last_layer, n_tab=n_tab, i=i))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, conv_builder, n_frames_last_layer=n_frames_last_layer,
                      input_last_layer=input_last_layer, n_tab=n_tab, i=i))

        return DoubleInputSequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _video_resnet(**kwargs):
    model = VideoResNet(**kwargs)

    return model


def ResNetTabAttention(input_size=(224, 224), n_frames=16, n_tab=6, **kwargs):
    n_frames_last_layer = max(n_frames // 8, 1)
    input_last_layer = (input_size[0] // 16, input_size[1] // 16)
    return _video_resnet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem,
                         n_frames_last_layer=n_frames_last_layer, input_last_layer=input_last_layer, n_tab=n_tab,
                         **kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 1, 16, 128, 128)
    tab = torch.randn(1, 1, 6)
    model = ResNetTabAttention(input_size=(x.shape[-2], x.shape[-1]), n_frames=x.shape[2], n_tab=tab.shape[-1])
    print(model)
    print(model(x, tab))
