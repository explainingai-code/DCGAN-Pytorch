import torch
import numpy as np
import torch.nn as nn


'''
Source code from 
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 
and 
https://arxiv.org/pdf/1511.06434.pdf

'''
latent_dim = 100
in_channels = [512, 256, 128, 64]
kernel_size = [4,4,4,4,4]
stride = [1,2,2,2,2]
padding = [0,1,1,1,1]
out_channels = 3


class Generator(nn.Module):
    def __init__(self,latent_dim = latent_dim, in_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, out_channels = out_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        self.main = nn.Sequential(
            ###
            nn.ConvTranspose2d(self.latent_dim,self.in_channels[0],self.kernel_size[0], self.stride[0], self.padding[0], bias=False),
            nn.BatchNorm2d(self.in_channels[0]),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(self.in_channels[0],self.in_channels[1],self.kernel_size[1], self.stride[1], self.padding[1], bias=False),
            nn.BatchNorm2d(self.in_channels[1]),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(self.in_channels[1],self.in_channels[2],self.kernel_size[2], self.stride[2], self.padding[2], bias=False),
            nn.BatchNorm2d(self.in_channels[2]),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(self.in_channels[2],self.in_channels[3],self.kernel_size[3], self.stride[3], self.padding[3], bias=False),
            nn.BatchNorm2d(self.in_channels[3]),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(self.in_channels[3],self.out_channels,self.kernel_size[4], self.stride[4], self.padding[4], bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, out_channels = out_channels):
        super(Discriminator, self).__init__()
        self.in_channels = np.flip(in_channels)
        self.kernel_size = np.flip(kernel_size)
        self.stride = np.flip(stride)
        self.padding = np.flip(padding)
        self.out_channels = out_channels

        self.main = nn.Sequential(
            ###
            nn.Conv2d(self.out_channels,self.in_channels[0],self.kernel_size[0], self.stride[0], self.padding[0], bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(self.in_channels[0],self.in_channels[1],self.kernel_size[1], self.stride[1], self.padding[1], bias=False),
            nn.BatchNorm2d(self.in_channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(self.in_channels[1],self.in_channels[2],self.kernel_size[2], self.stride[2], self.padding[2], bias=False),
            nn.BatchNorm2d(self.in_channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(self.in_channels[2],self.in_channels[3],self.kernel_size[3], self.stride[3], self.padding[3], bias=False),
            nn.BatchNorm2d(self.in_channels[3]),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(self.in_channels[3],1,self.kernel_size[4], self.stride[4], self.padding[4], bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    


        
















































'''Doubts:
- effect of bias=False
- effect of nn.ReLU(True) vs nn.ReLU()

'''