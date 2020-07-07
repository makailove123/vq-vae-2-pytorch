import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn
from vqvae_config import VqvaeConfig, VqVaeConfig2


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, config: VqvaeConfig):
        super().__init__()

        self.config = config
        in_channel = config.in_channel
        channel = config.channel
        n_res_block = config.n_res_block
        n_res_channel = config.n_res_channel
        embed_dim = config.embed_dim
        n_embed = config.n_embed
        decay = config.decay
        bottom_stride = config.bottom_stride

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=bottom_stride)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=bottom_stride,
        )

    def forward(self, input):
        quant_t, quant_b, diff, id_t, id_b = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff, id_t, id_b

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        #print(1, "enc_b", enc_b.shape, "enc_t", enc_t.shape)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        #print(2, "quant_t", quant_t.shape)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        #print(3, "quant_t", quant_t.shape, "diff_t", diff_t.shape, "id_t", id_t.shape)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        #print(4, "quant_t", quant_t.shape, "diff_t", diff_t.shape, "id_t", id_t.shape)
        dec_t = self.dec_t(quant_t)
        #print("4-1", "dec_t", dec_t.shape)
        enc_b = torch.cat([dec_t, enc_b], 1)
        #print(5, "dec_t", dec_t.shape, "enc_b", enc_b.shape)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        #print(6, "quant_b", quant_b.shape)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        #print(7, "quant_b", quant_b.shape, "diff_b", diff_b.shape, "id_b", id_b.shape)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        #print(8, "quant_b", quant_b.shape, "diff_b", diff_b.shape, "id_b", id_b.shape)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def transconv2x2(in_channels, out_channels):
    """upsampling by double"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def make_res_layer(in_channels, out_channels, stride):
    layers = []
    if stride != 1 or in_channels != out_channels:
        layers.append(conv3x3(in_channels, out_channels, stride))
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(ResBlock2(out_channels, out_channels))
    layers.append(ResBlock2(out_channels, out_channels))
    return nn.Sequential(*layers)


def make_transpose_res_layer(in_channels, out_channels):
    layers = [ResBlock2(in_channels, in_channels), ResBlock2(in_channels, in_channels),
              transconv2x2(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        cur_channels = 64
        self.conv1 = nn.Conv2d(3, cur_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(cur_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_res_layer(cur_channels, 64, 1)
        self.layer2 = make_res_layer(cur_channels, 128, 2)
        cur_channels = 128
        self.layer3 = make_res_layer(cur_channels, 256, 2)
        cur_channels = 256
        self.layer4 = make_res_layer(cur_channels, 512, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        bottom = self.layer3(out)
        top = self.layer4(bottom)

        return bottom, top


class Decoder2Top(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder2Top, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.layer1 = make_transpose_res_layer(512, 256)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        return out


class Decoder2Bottom(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder2Bottom, self).__init__()
        cur_channels = 256
        self.conv1 = nn.Conv2d(embed_dim, cur_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cur_channels)
        self.layer1 = make_transpose_res_layer(cur_channels, cur_channels // 2)
        cur_channels //= 2
        self.layer2 = make_transpose_res_layer(cur_channels, cur_channels // 2)
        cur_channels //= 2
        self.layer3 = make_transpose_res_layer(cur_channels, cur_channels)
        self.layer4 = make_transpose_res_layer(cur_channels, cur_channels)
        self.out_layer = conv1x1(cur_channels, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.out_layer(out)
        return out


class VqVae2(nn.Module):
    def __init__(self, config: VqVaeConfig2):
        super(VqVae2, self).__init__()

        self.config = config
        self.enc_top_channels = 512
        self.enc_bottom_channels = 256

        self.enc = Encoder2()
        self.quantize_conv_t = nn.Conv2d(self.enc_top_channels, config.top_embed_dim, 1)
        self.quantize_t = Quantize(config.top_embed_dim, config.top_n_embed)
        self.dec_t = Decoder2Top(config.top_embed_dim)

        self.quantize_conv_b = nn.Conv2d(self.enc_bottom_channels * 2, config.bottom_embed_dim, 1)
        self.quantize_b = Quantize(config.bottom_embed_dim, config.bottom_n_embed)
        self.upsample_t = transconv2x2(config.top_embed_dim, config.top_embed_dim)
        self.dec_b = Decoder2Bottom(config.bottom_embed_dim + config.top_embed_dim)

    def encode(self, input):
        enc_b, enc_t = self.enc(input)
        #print(1, "enc_b", enc_b.shape, "enc_t", enc_t.shape)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        #print(2, "quant_t", quant_t.shape)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        #print(3, "quant_t", quant_t.shape, "diff_t", diff_t.shape, "id_t", id_t.shape)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        #print(4, "quant_t", quant_t.shape, "diff_t", diff_t.shape, "id_t", id_t.shape)
        dec_t = self.dec_t(quant_t)
        #print("4-1", "dec_t", dec_t.shape)
        enc_b = torch.cat([dec_t, enc_b], 1)
        #print(5, "dec_t", dec_t.shape, "enc_b", enc_b.shape)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        #print(6, "quant_b", quant_b.shape)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        #print(7, "quant_b", quant_b.shape, "diff_b", diff_b.shape, "id_b", id_b.shape)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        #print(8, "quant_b", quant_b.shape, "diff_b", diff_b.shape, "id_b", id_b.shape)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        quant_t = self.upsample_t(quant_t)
        quant = torch.cat([quant_t, quant_b], 1)
        dec = self.dec_b(quant)
        #print(10, "dec", dec.shape)
        return dec

    def forward(self, input):
        quant_t, quant_b, diff, id_t, id_b = self.encode(input)
        #print(9, "quant_t", quant_t.shape, "quant_b", quant_b.shape, "diff", diff, "id_t", id_t.shape, "id_b", id_b.shape)
        dec = self.decode(quant_t, quant_b)

        return dec, diff, id_t, id_b

