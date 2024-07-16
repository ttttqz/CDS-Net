import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from thop import profile
from module.MDHA import MDHA
from module.DSRB import DSRB


class CDSNET(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=[32, 64, 128, 256]):

        super(CDSNET, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList()
        self.sspcab = nn.ModuleList()
        for feature in reversed(features):
            self.sspcab.append(DSRB(channels=feature))

        # Down part
        for feature in features:
            self.downs.append(MDHA(in_channels, feature))
            in_channels = feature

        # UP part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(MDHA(feature * 2, feature))

        self.bottleneck = nn.Sequential(MDHA(features[-1], features[-1] * 2), DSRB(channels=features[-1]*2))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # UP part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            skip_connection = self.sspcab[idx // 2](skip_connection)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

def test():
    x = torch.randn((1, 3, 320, 480)).cuda()  # batch size, channel,height,width

    model = CDSNET(in_channels=3, out_channels=1).cuda()
    # flops, params = profile(model, (x,))
    # print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()
