import torch
import torch.nn as nn



class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class mask_unet(nn.Module):
    def __init__(self):
        super(mask_unet, self).__init__()

        # Encoder path
        self.enc1_1 = conv_block(in_channel=256, out_channel=256)
        self.enc1_2 = conv_block(in_channel=256, out_channel=256)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = conv_block(in_channel=256, out_channel=512)
        self.enc2_2 = conv_block(in_channel=512, out_channel=512)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = conv_block(in_channel=512, out_channel=1024)

        # Decoder path
        self.dec3_1 = conv_block(in_channel=1024, out_channel=512)

        self.unpool2 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = conv_block(in_channel= 2 * 512, out_channel=512)
        self.dec2_1 = conv_block(in_channel=512, out_channel=256)

        self.unpool1 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = conv_block(in_channel=2 * 256, out_channel=256)
        self.dec1_1 = conv_block(in_channel=256, out_channel=256)

        self.fc = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0)

        # for name, param in self.named_parameters():
        #     if "weight" in name:
        #         nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, input):
        enc1_1 = self.enc1_1(input)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)

        dec3_1 = self.dec3_1(enc3_1)
        print(dec3_1.shape)
        unpool2 = self.unpool2(dec3_1)
        print(unpool2.shape)
        cat2 = torch.cat((enc2_2, unpool2), dim=1)
        print(cat2.shape)
        dec2_2 = self.dec2_2(cat2)
        print(dec2_2.shape)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((enc1_2, unpool1), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)

        return output
