import glob
import os
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default resize them images
    such that smaller dimension is 64 and then do centre cropping.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    def __init__(self, split, im_path, im_size=64, im_channels=3, im_ext='jpg'):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.images = self.load_images(im_path)
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        for fname in glob.glob(os.path.join(im_path, '*.{}'.format(self.im_ext))):
            ims.append(fname)
        print('Found {} images'.format(len(ims)))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor


# if __name__ == '__main__':
#     mnist = CelebDataset(im_path='/Users/tusharkumar/Downloads/img_align_celeba')
#     dataloader = DataLoader(mnist, batch_size=225)
#     for data in dataloader:
#         import torchvision
#         from torchvision.utils import make_grid
#
#         data = (data + 1) / 2
#         grid = make_grid(data, nrow=15)
#         img = torchvision.transforms.ToPILImage()(grid)
#
#         img.save('dataset_sample.png')
#         img.close()
#         exit()
