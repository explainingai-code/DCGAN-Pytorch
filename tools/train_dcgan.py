import torch
import os
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
import sys 
sys.path.append("./")
from torch.utils.data import DataLoader
from model.dcgans import Generator, Discriminator
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations used for creating
# and training GAN
LATENT_DIM = 100
IM_CHANNELS = 3
IM_PATH = "./data/img_align_celeba/"
# IM_EXT = 'png'
IM_SIZE = (64, 64)
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_SAMPLES = 225
NROWS = 15
N_workers = 4
####################################



def train():
    # Create the dataset
    dataset = torchvision.datasets.ImageFolder(root=IM_PATH, transform=transforms.Compose([
                               transforms.Resize(IM_SIZE),
                               transforms.CenterCrop(IM_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=N_workers)
    
    
    # Instantiate the model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.train()
    discriminator.train()
    
    # Specify training parameters
    optimizer_generator = Adam(generator.parameters(), lr=1E-4, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=1E-4, betas=(0.5, 0.999))
    
    # Criterion is bcewithlogits hence no sigmoid in discriminator
    criterion = torch.nn.BCELoss()

    # Run training
    steps = 0
    generated_sample_count = 0
    for epoch_idx in range(NUM_EPOCHS):
        generator_losses = []
        discriminator_losses = []
        mean_real_dis_preds = []
        mean_fake_dis_preds = []
        for im in tqdm(dataloader):
            real_ims = im[0].float().to(device)
            batch_size = real_ims.shape[0]
            
            # Optimize Discriminator
            optimizer_discriminator.zero_grad()
            fake_im_noise = torch.randn((batch_size, LATENT_DIM, 1,1), device=device)
            fake_ims = generator(fake_im_noise)
            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)
            
            disc_real_pred = discriminator(real_ims)
            disc_fake_pred = discriminator(fake_ims.detach())
            disc_real_loss = criterion(disc_real_pred.reshape(-1), real_label.reshape(-1))
            mean_real_dis_preds.append(torch.nn.Sigmoid()(disc_real_pred).mean().item())

            disc_fake_loss = criterion(disc_fake_pred.reshape(-1), fake_label.reshape(-1))
            mean_fake_dis_preds.append(torch.nn.Sigmoid()(disc_fake_pred).mean().item())
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            optimizer_discriminator.step()
            ########################
            
            # Optimize Generator
            optimizer_generator.zero_grad()
            fake_im_noise = torch.randn((batch_size, LATENT_DIM, 1,1), device=device)
            fake_ims = generator(fake_im_noise)
            disc_fake_pred = discriminator(fake_ims)
            gen_fake_loss = criterion(disc_fake_pred.reshape(-1), real_label.reshape(-1))
            gen_fake_loss.backward()
            optimizer_generator.step()
            ########################
            
            generator_losses.append(gen_fake_loss.item())
            discriminator_losses.append(disc_loss.item())
            
            # Save samples
            if steps % 50 == 0:
                with torch.no_grad():
                    generator.eval()
                    infer(generated_sample_count, generator)
                    generated_sample_count += 1
                    generator.train()
            #############
            steps += 1
        print('Finished epoch:{} | Generator Loss : {:.4f} | Discriminator Loss : {:.4f} | '
              'Discriminator real pred : {:.4f} | Discriminator fake pred : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(generator_losses),
            np.mean(discriminator_losses),
            np.mean(mean_real_dis_preds),
            np.mean(mean_fake_dis_preds),
        ))
        torch.save(generator.state_dict(), 'generator_ckpt.pth')
        torch.save(discriminator.state_dict(), 'discriminator_ckpt.pth')
    
    print('Done Training ...')


def infer(generated_sample_count, generator):
    r"""
    Method to save the generated samples
    :param generated_sample_count: Filename to save the output with
    :param generator: Generator model with trained parameters
    :return:
    """
    fake_im_noise = torch.randn((NUM_SAMPLES, LATENT_DIM, 1, 1), device=device)
    fake_ims = generator(fake_im_noise)
    ims = torch.clamp(fake_ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=NROWS)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists('samples'):
        os.mkdir('samples')
    img.save('samples/{}.png'.format(generated_sample_count))


if __name__ == '__main__':
    train()
