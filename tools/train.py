import torch
import os
import argparse
import yaml
import numpy as np
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
from model.dcgan import Generator, Discriminator
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celeb': CelebDataset
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                im_ext=dataset_config['im_ext'])
    mnist_loader = DataLoader(im_dataset, batch_size=train_config['batch_size'], shuffle=True)
    
    # Instantiate the model
    generator = Generator(latent_dim=model_config['latent_dim'],
                          im_size=dataset_config['im_size'],
                          im_channels=dataset_config['im_channels'],
                          conv_channels=model_config['generator_channels'],
                          kernels=model_config['generator_kernels'],
                          strides=model_config['generator_strides'],
                          paddings=model_config['generator_paddings'],
                          output_paddings=model_config['generator_output_paddings']).to(device)

    discriminator = Discriminator(im_size=dataset_config['im_size'],
                                  im_channels=dataset_config['im_channels'],
                                  conv_channels=model_config['discriminator_channels'],
                                  kernels=model_config['discriminator_kernels'],
                                  strides=model_config['discriminator_strides'],
                                  paddings=model_config['discriminator_paddings']).to(device)
    generator.train()
    discriminator.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   '_'.join(['generator', train_config['ckpt_name']]))):
        if os.path.exists(os.path.join(train_config['task_name'],
                                       '_'.join(['discriminator', train_config['ckpt_name']]))):
            
            # Load checkpoint ONLY if both generator and discriminator are present
            print('Loading checkpoints as found them')
            generator.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                              '_'.join(['generator', train_config['ckpt_name']])),
                                                 map_location=device))
            discriminator.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                  '_'.join(['discriminator',
                                                                            train_config['ckpt_name']])),
                                                     map_location=device))
        
    # Specify training parameters
    optimizer_generator = Adam(generator.parameters(), lr=train_config['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=train_config['lr'], betas=(0.5, 0.999))
    
    # Criterion is bcewithlogits hence no sigmoid in discriminator
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Run training
    steps = 0
    generated_sample_count = 0
    
    for epoch_idx in range(train_config['num_epochs']):
        generator_losses = []
        discriminator_losses = []
        mean_real_dis_preds = []
        mean_fake_dis_preds = []
        for im in tqdm(mnist_loader):
            real_ims = im.float().to(device)
            batch_size = real_ims.shape[0]
            
            # Optimize Discriminator
            optimizer_discriminator.zero_grad()
            fake_im_noise = torch.randn((batch_size, model_config['latent_dim']), device=device)
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
            fake_im_noise = torch.randn((batch_size, model_config['latent_dim']), device=device)
            fake_ims = generator(fake_im_noise)
            disc_fake_pred = discriminator(fake_ims)
            gen_fake_loss = criterion(disc_fake_pred.reshape(-1), real_label.reshape(-1))
            gen_fake_loss.backward()
            optimizer_generator.step()
            ########################
            
            generator_losses.append(gen_fake_loss.item())
            discriminator_losses.append(disc_loss.item())
            
            # Save samples
            if steps % train_config['save_sample_steps'] == 0:
                generator.eval()
                infer(generated_sample_count, generator,
                      train_config, model_config)
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
        torch.save(generator.state_dict(), os.path.join(train_config['task_name'],
                                                        '_'.join(['generator', train_config['ckpt_name']])))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            '_'.join(['discriminator', train_config['ckpt_name']])))
    
    print('Done Training ...')


def infer(generated_sample_count, generator, train_config, model_config):
    r"""
    Method to save the generated samples
    :param generated_sample_count: Filename to save the output with
    :param generator: Generator model with trained parameters
    :param train_config: Training configuration picked up from yaml
    :param model_config: Model configuration picked up from yaml
    """
    
    with torch.no_grad():
        fake_im_noise = torch.randn((train_config['num_samples'], model_config['latent_dim']), device=device)
        fake_ims = generator(fake_im_noise)
        fake_ims = (fake_ims + 1) / 2
        grid = make_grid(fake_ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', '{}.png'.format(generated_sample_count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dcgan training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
