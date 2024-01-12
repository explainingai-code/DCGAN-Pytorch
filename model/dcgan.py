import torch
import torch.nn as nn


class Generator(nn.Module):
    r"""
    Generator for this gan is list of layers where each layer has the following:
    1. Conv Transpose Layer
    2. BatchNorm
    3. Activation(Tanh for last layer else LeakyRELU)
    The conv layers progressively increase dimension
    from LATENT_DIMx1x1 to IMG_CHANNELSxIMG_HxIMG_W
    """
    def __init__(self, latent_dim, im_size, im_channels,
                 conv_channels, kernels, strides, paddings,
                 output_paddings):
        super().__init__()
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.im_channels = im_channels
        
        activation = nn.ReLU()
        layers_dim = [self.latent_dim] + conv_channels + [self.im_channels]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(layers_dim[i], layers_dim[i + 1],
                                   kernel_size=kernels[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   bias=False),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim, 1, 1)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.im_channels, self.im_size, self.im_size)
        return out


class Discriminator(nn.Module):
    r"""
    Discriminator mimicks the design of generator
    only reduces dimensions progressive rather than increasing
    using strided convolutions.
    From IMG_CHANNELSxIMG_HxIMG_W it reduces all the way to 1 where
    the last value is the probability discriminator thinks that
    given image is real(closer to 1 if real else closer to 0)
    """
    
    def __init__(self, im_size, im_channels,
                 conv_channels, kernels, strides, paddings):
        super().__init__()
        self.img_size = im_size
        self.im_channels = im_channels
        activation = nn.LeakyReLU()
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i]),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.reshape(x.size(0))


if __name__ == '__main__':
    x = torch.randn((2, 100))
    out = Generator()(x)
    print(out.shape)
    prob = Discriminator()(out)
