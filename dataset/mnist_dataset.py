import glob
import os
import random
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split, im_path, im_size=28, im_channels=1, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_ext = im_ext
        self.im_channels = im_channels
        self.images, self.labels = self.load_images(im_path)
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up.
        Assumes the im_path location has subdirectories
        and inside those sub-folders are present the images for
        each class
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                # labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        im.close()
        
        if self.im_channels == 3:
            a = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
            b = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
            c = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
            im_tensor = torch.cat([a, b, c], dim=0)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
