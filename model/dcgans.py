import torch
import torch.nn as nn


'''
Source code from 
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 
and 
https://arxiv.org/pdf/1511.06434.pdf

'''
class Generator(nn.Module):
    def __init__(self,latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            ###
            nn.ConvTranspose2d(self.latent_dim,1024,4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(1024,512,5, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(512,256,5, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(256,128,5, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ###
            nn.ConvTranspose2d(128,3,5, 2, 1, bias=False),
            nn.Tanh()

        )
    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            ###
            nn.Conv2d(3,128,5, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(128,256,5, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(256,512,5, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(512,1024,5, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            ###
            nn.Conv2d(1024,1,4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    
    


        
















































'''Doubts:
- effect of bias=False
- effect of nn.ReLU(True) vs nn.ReLU()

'''