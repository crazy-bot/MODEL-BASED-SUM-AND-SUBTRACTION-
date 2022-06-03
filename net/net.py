import torch.nn as nn
import torch.nn.functional as F
import torch

from .util import ConvBlock, LinearBlock, UpsampleBlock

class NaiveModel(nn.Module):
    def __init__(self, in_channel=2, out_channel=10) -> None:
        super().__init__()

        conv1 = ConvBlock(in_channel, 16, 3, 1, 0) # 26*26*16
        conv2 = ConvBlock(16, 32, 3, 2, 0) # 12*12*32
        conv3 = ConvBlock(32, 64, 3, 2, 0) # 5*5*64 = 1600

        fc1 = LinearBlock(5*5*64, 800)
        fc2 = LinearBlock(800, 400)
        fc3 = LinearBlock(400, out_channel)

        self.featnet =  nn.Sequential(conv1, conv2, conv3,)
        self.classifier =  nn.Sequential(fc1, fc2, fc3)

    def forward(self, input1, input2):

        input = torch.cat((input1, input2), dim=1)
        x = self.featnet(input)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x



class TwoTowerModel(nn.Module):
    def __init__(self, in_channel=1, out_channel=10) -> None:
        super().__init__()

        conv1 = ConvBlock(in_channel, 16, 3, 1, 0) # 16*26*26
        conv2 = ConvBlock(16, 32, 3, 2, 0) # 32*12*12
        conv3 = ConvBlock(32, 64, 3, 2, 0) # 64*5*5 = 1600

        self.featnet =  nn.Sequential(conv1, conv2, conv3)

        conv4 = ConvBlock(128, 64, 3, 1, 0) # 64*3*3
        fc1 = LinearBlock(3*3*64, 3*3*32)
        fc2 = LinearBlock(3*3*32, 3*3*16)
        fc3 = LinearBlock(3*3*16, out_channel)

        self.classifier =  nn.Sequential(conv4, nn.Flatten(start_dim=1), fc1, fc2, fc3)

    def forward(self, input1, input2):

        feat1 = self.featnet(input1)
        feat2 = self.featnet(input2)
        
        x = torch.cat((feat1, feat2), dim=1)
        
        out = self.classifier(x)

        return out




class Encoder(nn.Module):
    
    def __init__(self, input_dim, encoded_dim=128 ):
        super().__init__()
        
        self.net = nn.Sequential(
            ConvBlock(input_dim, 8, 3, stride=2, pad=1), 
            ConvBlock(8, 16, 3, stride=2, pad=1),
            ConvBlock(16, 32, 3, stride=2, pad=0),
            nn.Flatten(start_dim=1),
            LinearBlock(3 * 3 * 32, encoded_dim)

        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, input_dim, encoded_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            LinearBlock(encoded_dim, 3 * 3 * 32),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            UpsampleBlock(32, 16, 3, stride=2), 
            UpsampleBlock(16, 8, 3, stride=2, pad=1, out_pad=1),
            UpsampleBlock(8, input_dim, 3, stride=2, pad=1, out_pad=1)
        
        )
        
    def forward(self, x):
        x = self.net(x)
        #x = torch.sigmoid(x)
        return x

class AutoEncoderModel(nn.Module):
    def __init__(self, input_dim, encoded_dim=128, out_channel=10) -> None:
        super().__init__()
        
        self.encoder = Encoder(input_dim, encoded_dim=encoded_dim)
        self.decoder = Decoder(input_dim, encoded_dim=encoded_dim)

        fc1 = LinearBlock(encoded_dim*2, encoded_dim)
        fc2 = LinearBlock(encoded_dim, encoded_dim//2)
        fc3 = LinearBlock(encoded_dim//2, out_channel)

        self.classifier =  nn.Sequential(fc1, fc2, fc3)

    def forward(self, input1, input2):

        latent1 = self.encoder(input1)
        x1 = self.decoder(latent1)
        latent2 = self.encoder(input2)
        x2 = self.decoder(latent2)

        x = torch.cat((latent1, latent2), dim=1)
        x = torch.flatten(x, start_dim=1)
        #import pdb; pdb.set_trace()
        out = self.classifier(x)

        return x1, x2, out


class TwoTowerModelExtended(nn.Module):
    def __init__(self, in_channel=1, out_channel=10) -> None:
        super().__init__()

        conv1 = ConvBlock(in_channel, 16, 3, 1, 0) # 16*26*26
        conv2 = ConvBlock(16, 32, 3, 2, 0) # 32*12*12
        conv3 = ConvBlock(32, 64, 3, 2, 0) # 64*5*5 = 1600

        self.featnet =  nn.Sequential(conv1, conv2, conv3)

        self.conv4 = ConvBlock(128, 64, 3, 1, 0) # 64*3*3
        fc1 = LinearBlock(3*3*64+1, 3*3*32)
        fc2 = LinearBlock(3*3*32, 3*3*16)
        fc3 = LinearBlock(3*3*16, out_channel)

        self.classifier =  nn.Sequential(fc1, fc2, fc3)

    def forward(self, input1, input2, flag):

        feat1 = self.featnet(input1)
        feat2 = self.featnet(input2)
        #import pdb; pdb.set_trace()

        x = torch.cat((feat1, feat2), dim=1)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, flag), dim=1)

        out = self.classifier(x)

        return out


class AutoEncoderModelExtended(nn.Module):
    def __init__(self, input_dim, encoded_dim=128, out_channel=10) -> None:
        super().__init__()
        
        self.encoder = Encoder(input_dim, encoded_dim=encoded_dim)
        self.decoder = Decoder(input_dim, encoded_dim=encoded_dim)

        fc1 = LinearBlock(encoded_dim, encoded_dim)
        fc2 = LinearBlock(encoded_dim, encoded_dim//2)
        fc3 = LinearBlock(encoded_dim//2, out_channel)

        self.classifier =  nn.Sequential(fc1, fc2, fc3)

    def forward(self, input1, input2, flag):

        latent1 = self.encoder(input1)
        x1 = self.decoder(latent1)
        latent2 = self.encoder(input2)
        x2 = self.decoder(latent2)

        
        if flag:
            x = torch.add(latent1, latent2)
        else:
            x = torch.sub(latent1, latent2)

        x = torch.flatten(x, start_dim=1)
        #import pdb; pdb.set_trace()
        out = self.classifier(x)

        return x1, x2, out