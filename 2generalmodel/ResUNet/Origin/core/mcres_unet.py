import torch
import torch.nn as nn
from core.modules import ResidualConv, Upsample


class MCResUnet(nn.Module):
    def __init__(self, channel_x0, channel_i0, out_channel, filters=[64, 128, 256, 512]):
        super(MCResUnet, self).__init__()
        # x0
        self.input_layer_x0 = nn.Sequential(
            nn.Conv2d(channel_x0, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip_x0 = nn.Sequential(
            nn.Conv2d(channel_x0, filters[0], kernel_size=3, padding=1)
        )

        # i0
        self.input_layer_i0 = nn.Sequential(
            nn.Conv2d(channel_i0, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip_i0 = nn.Sequential(
            nn.Conv2d(channel_i0, filters[0], kernel_size=3, padding=1)
        )
        ## bridge adding
        self.channelbridge = ResidualConv(filters[3] + filters[3], filters[3], 1, 1)

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x0, i0):
        # x0 Encode
        x1 = self.input_layer_x0(x0) + self.input_skip_x0(x0)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # i0 Encode
        i1 = self.input_layer_i0(i0) + self.input_skip_i0(i0)
        i2 = self.residual_conv_1(i1)
        i3 = self.residual_conv_2(i2)

        # Bridge
        x4 = self.bridge(x3)
        i4 = self.bridge(i3)

        # Decode
        x4 = torch.cat([x4, i4], dim=1) # output channels = filters[3]+filters[3]
        x4 = self.channelbridge(x4)  # output channels = filters[3]

        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, i3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, i2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, i1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
