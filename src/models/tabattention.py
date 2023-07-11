import torch.nn as nn
import torch
import torch.nn.functional as F
import math


# _______________ CBAM from: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py ___________________

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=False, n_tab=6,
                 tabular_branch=True):
        super(ChannelGate, self).__init__()
        self.tabattention = tabattention
        self.n_tab = n_tab
        self.gate_channels = gate_channels
        self.tabular_branch = tabular_branch
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            if self.tabular_branch:
                self.tab_embedding = nn.Identity()
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, gate_channels // reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(gate_channels // reduction_ratio, gate_channels)
                )

    def forward(self, x, tab=None):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            elif pool_type == 'tab':
                embedded = self.tab_embedding(tab)
                embedded = torch.reshape(embedded, (-1, self.gate_channels))
                pool = self.mlp(embedded)
                channel_att_raw = pool

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class TemporalMHSA(nn.Module):
    def __init__(self, input_dim=2, seq_len=16, heads=2):
        super(TemporalMHSA, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = 4
        self.head_dim = self.embedding_dim // heads
        self.heads = heads
        self.qkv = nn.Linear(self.input_dim, self.embedding_dim * 3)
        self.rel = nn.Parameter(torch.randn([1, 1, seq_len, 1]), requires_grad=True)
        self.o_proj = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        d_k = q.size()[-1]
        k = k + self.rel.expand_as(k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)  # [Batch, SeqLen, EmbeddingDim]
        x_out = self.o_proj(values)

        return x_out


class TemporalGate(nn.Module):
    def __init__(self, gate_frames, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=False, n_tab=6,
                 temporal_mhsa=False, tabular_branch=True):
        super(TemporalGate, self).__init__()
        self.tabattention = tabattention
        self.tabular_branch = tabular_branch
        self.n_tab = n_tab
        self.gate_frames = gate_frames
        self.temporal_mhsa = temporal_mhsa
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_frames, gate_frames // 2),
            nn.ReLU(),
            nn.Linear(gate_frames // 2, gate_frames)
        )
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            if self.tabular_branch:
                self.tab_embedding = nn.Sequential(nn.Linear(n_tab, gate_frames), nn.ReLU())
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, gate_frames // 2),
                    nn.ReLU(),
                    nn.Linear(gate_frames // 2, gate_frames)
                )
        if self.temporal_mhsa:
            if tabattention:
                self.mhsa = TemporalMHSA(input_dim=3, seq_len=self.gate_frames)
            else:
                self.mhsa = TemporalMHSA(input_dim=2, seq_len=self.gate_frames)

    def forward(self, x, tab=None):
        if not self.temporal_mhsa:
            channel_att_sum = None
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                            stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                            stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == 'tab':
                    embedded = self.tab_embedding(tab)
                    embedded = torch.reshape(embedded, (-1, self.gate_frames))
                    pool = self.mlp(embedded)
                    channel_att_raw = pool

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

            scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
            return x * scale
        else:
            avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                    stride=(x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)
            max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                    stride=(x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)
            if self.tabattention:
                embedded = self.tab_embedding(tab)
                tab_embedded = torch.reshape(embedded, (-1, self.gate_frames, 1))
                concatenated = torch.cat((avg_pool, max_pool, tab_embedded), dim=2)
            else:
                concatenated = torch.cat((avg_pool, max_pool), dim=2)

            scale = torch.sigmoid(self.mhsa(concatenated)).unsqueeze(2).unsqueeze(3).expand_as(x)

            return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, tabattention=False, n_tab=6, input_size=(8, 8), tabular_branch=True):
        super(SpatialGate, self).__init__()
        self.tabattention = tabattention
        self.tabular_branch = tabular_branch
        self.n_tab = n_tab
        self.input_size = input_size
        kernel_size = 7
        self.compress = ChannelPool()
        in_planes = 3 if tabattention else 2
        self.spatial = BasicConv(in_planes, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        if self.tabattention:
            if self.tabular_branch:
                self.tab_embedding = nn.Sequential(nn.Linear(n_tab, input_size[0] * input_size[1]), nn.ReLU())
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, input_size[0] * input_size[1] // 2),
                    nn.ReLU(),
                    nn.Linear(input_size[0] * input_size[1] // 2, input_size[0] * input_size[1])
                )

    def forward(self, x, tab=None):
        x_compress = self.compress(x)
        if self.tabattention:
            embedded = self.tab_embedding(tab)
            embedded = torch.reshape(embedded, (-1, 1, self.input_size[0], self.input_size[1]))
            x_compress = torch.cat((x_compress, embedded), dim=1)

        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    # CBAM enhanced with TAM and tabular embeddings
    def __init__(self, gate_channels, frame_channels, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=False,
                 n_tab=6, temporal_mhsa=False, input_size=(8, 8), temporal_attention=False, tabular_branch=True,
                 cam_sam=False):
        super(CBAM, self).__init__()
        if tabular_branch:
            n_tab = gate_channels
        self.n_tab = n_tab
        self.tabattention = tabattention
        self.temporal_attention = temporal_attention
        self.cam_sam = cam_sam
        if self.cam_sam:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, tabattention, n_tab,
                                           tabular_branch=tabular_branch)
            self.SpatialGate = SpatialGate(tabattention, n_tab, input_size, tabular_branch=tabular_branch)
        if temporal_attention:
            self.TemporalGate = TemporalGate(frame_channels, tabattention=tabattention, n_tab=n_tab,
                                             temporal_mhsa=temporal_mhsa, tabular_branch=tabular_branch)

    def forward(self, x, tab):
        b, c, f, h, w = x.shape
        x_in = torch.permute(x, (0, 2, 1, 3, 4))
        x_in = torch.reshape(x_in, (b * f, c, h, w))
        if self.tabattention:
            tab_rep = tab.repeat(f, 1, 1)
        else:
            tab_rep = None

        if self.cam_sam:
            x_out = self.ChannelGate(x_in, tab_rep)
            x_out = self.SpatialGate(x_out, tab_rep)
        else:
            x_out = x_in

        x_out = torch.reshape(x_out, (b, f, c, h, w))

        if self.temporal_attention:
            x_out = self.TemporalGate(x_out, tab)

        x_out = torch.permute(x_out, (0, 2, 1, 3, 4))

        return x_out


# _______________ CBAM ____________________________________________________________________

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
                 input_last_layer=(7, 7), cbam=False, tabattention=False, n_tab=6, i=1, temporal_mhsa=False,
                 temporal_attention=False, tabular_branch=True, cam_sam=False):

        self.cbam = cbam

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        if not cbam:
            self.conv2 = nn.Sequential(
                conv_builder(planes, planes, midplanes),
                nn.BatchNorm3d(planes)
            )
        else:
            f_channels = (n_frames_last_layer * (2 ** (4 - i)))
            input_size = (input_last_layer[0] * (2 ** (4 - i)), input_last_layer[0] * (2 ** (4 - i)))
            self.cbam_layer = CBAM(gate_channels=planes, frame_channels=f_channels, tabattention=tabattention,
                                   n_tab=n_tab, temporal_mhsa=temporal_mhsa, input_size=input_size,
                                   temporal_attention=temporal_attention, tabular_branch=tabular_branch,
                                   cam_sam=cam_sam)
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
        if self.cbam:
            out = self.cbam_layer(out, tab)

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
                 cbam=False, tabattention=False, n_frames_last_layer=2, input_last_layer=(7, 7), n_tab=6,
                 temporal_mhsa=False, temporal_attention=False, tabular_branch=True, cam_sam=False):
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

        self.tabular_branch = tabular_branch

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1, cbam=cbam,
                                       tabattention=tabattention, n_frames_last_layer=n_frames_last_layer,
                                       input_last_layer=input_last_layer,
                                       n_tab=n_tab, i=1, temporal_mhsa=temporal_mhsa,
                                       temporal_attention=temporal_attention, tabular_branch=tabular_branch,
                                       cam_sam=cam_sam)

        if not head:
            self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2, cbam=cbam,
                                           tabattention=tabattention, n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=2,
                                           temporal_mhsa=temporal_mhsa, temporal_attention=temporal_attention,
                                           tabular_branch=tabular_branch, cam_sam=cam_sam)
            self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, cbam=cbam,
                                           tabattention=tabattention, n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=3,
                                           temporal_mhsa=temporal_mhsa, temporal_attention=temporal_attention,
                                           tabular_branch=tabular_branch, cam_sam=cam_sam)
            self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, cbam=cbam,
                                           tabattention=tabattention, n_frames_last_layer=n_frames_last_layer,
                                           input_last_layer=input_last_layer, n_tab=n_tab, i=4,
                                           temporal_mhsa=temporal_mhsa, temporal_attention=temporal_attention,
                                           tabular_branch=tabular_branch, cam_sam=cam_sam)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc_numerical = nn.Linear(6, 1)
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, num_classes)
            )

            if self.tabular_branch:
                self.tab_branch1 = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU())
                self.tab_branch2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
                self.tab_branch3 = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
                self.tab_branch4 = nn.Sequential(nn.Linear(256, 512), nn.ReLU())

        # init weights
        self._initialize_weights()

    def forward(self, x, tab=None):
        x = self.stem(x)

        if self.tabular_branch:
            tab = self.tab_branch1(tab)

        x = self.layer1(x, tab)
        if self.head:
            return x

        if self.tabular_branch:
            tab = self.tab_branch2(tab)

        x = self.layer2(x, tab)

        if self.tabular_branch:
            tab = self.tab_branch3(tab)
        x = self.layer3(x, tab)

        if self.tabular_branch:
            tab = self.tab_branch4(tab)
        x = self.layer4(x, tab)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)

        x = self.fc(x)
        x = x.flatten()
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, n_frames_last_layer=2,
                    input_last_layer=(7, 7), cbam=False, tabattention=False, n_tab=6, i=1, temporal_mhsa=False,
                    temporal_attention=False, tabular_branch=True, cam_sam=False):
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
            block(self.inplanes, planes, conv_builder, stride, downsample, cbam=cbam, tabattention=tabattention,
                  n_frames_last_layer=n_frames_last_layer,
                  input_last_layer=input_last_layer, n_tab=n_tab, i=i, temporal_mhsa=temporal_mhsa,
                  temporal_attention=temporal_attention, tabular_branch=tabular_branch, cam_sam=cam_sam))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, conv_builder, n_frames_last_layer=n_frames_last_layer,
                      input_last_layer=input_last_layer, cbam=cbam, tabattention=tabattention, n_tab=n_tab, i=i,
                      temporal_mhsa=temporal_mhsa, temporal_attention=temporal_attention,
                      tabular_branch=tabular_branch, cam_sam=cam_sam))

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


def TabAttention(cbam=False, tabattention=False, n_frames=16, input_size=(224, 224),
                 n_tab=6, temporal_mhsa=False, temporal_attention=False, tabular_branch=True, cam_sam=False, **kwargs):
    n_frames_last_layer = max(n_frames // 8, 1)
    input_last_layer = (input_size[0] // 16, input_size[1] // 16)
    return _video_resnet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2], stem=BasicStem, cbam=cbam, tabattention=tabattention,
                         n_frames_last_layer=n_frames_last_layer, input_last_layer=input_last_layer, n_tab=n_tab,
                         temporal_mhsa=temporal_mhsa, temporal_attention=temporal_attention,
                         tabular_branch=tabular_branch, cam_sam=cam_sam,
                         **kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 1, 16, 128, 128)
    tab = torch.randn(1, 1, 6)
    model = TabAttention(cbam=True, tabattention=True, input_size=(x.shape[-2], x.shape[-1]), temporal_mhsa=True,
                         temporal_attention=True, tabular_branch=False, cam_sam=True)
    print(model)
    print(model.forward(x, tab))
