import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, down=False, up=False, n_frame=4):
        super(ResBlock, self).__init__()
        assert not (down and up)

        self.scale = None
        if down:
            self.scale = nn.Conv2d(in_c, in_c, (3, 3), stride=(2, 2), padding=(1, 1))
        if up:
            self.scale = nn.ConvTranspose2d(in_c, in_c, (4, 4), stride=(2, 2), padding=(1, 1))

        self.covn1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.covn2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, (3, 3), stride=(1, 1), padding=(1, 1), groups=n_frame),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.covn3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        if in_c != out_c:
            self.residule = nn.Conv2d(in_c, out_c, (1, 1), stride=(1, 1), padding=(0, 0))
        else:
            self.residule = nn.Identity()

    def forward(self, x):
        if self.scale is not None:
            x = self.scale(x)
        h = self.covn1(x)
        h = self.covn2(h)
        h = self.covn3(h)
        h = h + self.residule(x)
        return h


class UNet(nn.Module):

    def __init__(self, num_classes: int, in_c: int, base_c: int, n_frame: int):
        super(UNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, base_c, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(base_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_c, base_c, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(base_c),
            nn.ReLU()
        )

        self.conv31 = ResBlock(base_c, base_c * 2, down=True)
        self.conv32 = ResBlock(base_c * 2, base_c * 2)
        self.conv33 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 2, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU()
        )

        self.conv41 = ResBlock(base_c * 2, base_c * 2, down=True)
        self.conv42 = ResBlock(base_c * 2, base_c * 2)
        self.conv43 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 2, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU()
        )
        
        self.conv51 = ResBlock(base_c * 2, base_c * 4, down=True)
        self.conv52 = ResBlock(base_c * 4, base_c * 4)
        self.conv53 = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c * 4, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c * 4),
            nn.ReLU()
        )
        
        self.dconv51 = ResBlock(base_c * 4, base_c * 4, up=True)
        self.dconv52 = ResBlock(base_c * 4, base_c * 2)
        self.dconv53 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 2, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU()
        )

        self.dconv41 = ResBlock(base_c * 2 + base_c * 2, base_c * 2, up=True)
        self.dconv42 = ResBlock(base_c * 2, base_c * 2)
        self.dconv43 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 2, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU()
        )

        self.dconv31 = ResBlock(base_c * 2 + base_c * 2, base_c * 2, up=True)
        self.dconv32 = ResBlock(base_c * 2, base_c)
        self.dconv33 = nn.Sequential(
            nn.Conv2d(base_c, base_c, (1, 1), stride=(1, 1), padding=(0, 0), groups=n_frame),
            nn.BatchNorm2d(base_c),
            nn.ReLU()
        )

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(base_c, base_c, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(base_c),
            nn.ReLU()
        )
        self.conv_out2 = nn.Sequential(
            nn.Conv2d(base_c, num_classes, (1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        l3 = self.conv31(x)
        l3 = self.conv32(l3)
        l3 = self.conv33(l3)

        l4 = self.conv41(l3)
        l4 = self.conv42(l4)
        l4 = self.conv43(l4)

        l5 = self.conv51(l4)
        l5 = self.conv52(l5)
        l5 = self.conv53(l5)

        l5 = self.dconv51(l5)
        l5 = self.dconv52(l5)
        _l5 = self.dconv53(l5)

        _l4 = torch.cat((_l5, l4), dim=1)
        _l4 = self.dconv41(_l4)
        _l4 = self.dconv42(_l4)
        _l4 = self.dconv43(_l4)

        _l3 = torch.cat((_l4, l3), dim=1)
        _l3 = self.dconv31(_l3)
        _l3 = self.dconv32(_l3)
        _l3 = self.dconv33(_l3)

        x = self.conv_out1(_l3)
        x = self.conv_out2(x)

        return x

if __name__ == "__main__":
    model = UNet(1024, 12, 64, 4)
    i = torch.randn(4, 12, 320, 640)
    out = model(i)
