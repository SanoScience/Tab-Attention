import torch.nn as nn
import torch

__all__ = ['r3d_18']


class MHSA3D(nn.Module):
    def __init__(self, n_dims, n_frames=2, width=14, height=14, heads=4):
        super(MHSA3D, self).__init__()
        self.scale = (n_dims // heads) ** -0.5
        self.heads = heads

        self.query = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)
        self.key = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                             bias=False)
        self.value = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, width, 1]), requires_grad=True)
        self.rel_t = nn.Parameter(torch.randn([1, heads, n_dims // heads, n_frames, 1, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, F, height, width = x.size()
        q = self.query(x).view(n_batch, self.heads, F, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, F, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, F, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 2, 4, 3), k)
        content_rel_pos = self.rel_h + self.rel_w + self.rel_t
        content_position = content_rel_pos.view(1, self.heads, C // self.heads, F, -1).permute(0, 1, 3, 4, 2)
        content_position2 = torch.matmul(content_position, q)

        energy = (content_content + content_position2) * self.scale
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 2, 4, 3))
        out = out.view(n_batch, C, F, height, width)

        return out


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

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, mhsa=False, n_frames_last_layer=2,
                 input_last_layer=(7, 7)):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        if not mhsa:
            self.conv2 = nn.Sequential(
                conv_builder(planes, planes, midplanes),
                nn.BatchNorm3d(planes)
            )
        else:
            self.conv2 = nn.Sequential(MHSA3D(planes, n_frames=n_frames_last_layer, width=input_last_layer[1],
                                              height=input_last_layer[0]), nn.BatchNorm3d(planes))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, y):
        residual = x

        out = self.conv1(x)

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

    def forward(self, x, y):
        for l in self.layers:
            x = l(x, y)
        return x


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, stem, num_classes=1, zero_init_residual=False, head=False,
                 msha=False, n_frames_last_layer=2, input_last_layer=(7, 7)):
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

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        if not head:
            self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, mhsa=msha,
                                           n_frames_last_layer=n_frames_last_layer, input_last_layer=input_last_layer)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc_numerical = nn.Linear(6, 1)
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, num_classes)
            )

        # init weights
        self._initialize_weights()

    def forward(self, x, y):
        x = self.stem(x)
        x = self.layer1(x, None)
        if self.head:
            return x
        x = self.layer2(x, None)
        x = self.layer3(x, None)
        x = self.layer4(x, y)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        # y = y.flatten(1)
        # x = torch.cat((x, y), dim=1)
        x = self.fc(x)
        x = x.flatten()
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, mhsa=False, n_frames_last_layer=2,
                    input_last_layer=(7, 7)):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, mhsa=mhsa,
                            n_frames_last_layer=n_frames_last_layer,
                            input_last_layer=input_last_layer))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, conv_builder, mhsa=mhsa, n_frames_last_layer=n_frames_last_layer,
                      input_last_layer=input_last_layer))

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
                if m.bias:
                    nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    return model


def r3d_18(pretrained=False, progress=True, msha=False, n_frames=16, input_size=(224, 224), **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """
    n_frames_last_layer = max(n_frames // 8, 1)
    input_last_layer = (input_size[0] // 16, input_size[1] // 16)
    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, msha=msha, n_frames_last_layer=n_frames_last_layer,
                         input_last_layer=input_last_layer, **kwargs)



if __name__ == "__main__":
    x = torch.randn(2, 1, 16, 224, 224)
    y = torch.randn(2, 1, 6)
    model = r3d_18()
    print(model.forward(x, y))
