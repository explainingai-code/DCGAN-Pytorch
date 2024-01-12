DCGAN Implementation in PyTorch
========

This repository implements training and inference methods of DCGAN on mnist as well as celeb faces dataset

[Paper](https://arxiv.org/pdf/1511.06434.pdf) </br>
[Video on DCGAN](https://youtu.be/672YP9k6Xws) </br>

## Sample Output after training on MNIST
<img src="https://github.com/explainingai-code/GANs-Pytorch/assets/144267687/4e1fd994-6ec0-4e21-aeee-6b054e72ddab" width="200">
<img src="https://github.com/explainingai-code/GANs-Pytorch/assets/144267687/f4bbdafa-a8e2-4a8f-b063-4a4bc00c76fa" width="200">

## Sample Output after training on CelebFaces Dataset

## Data preparation
For setting up the mnist dataset:
Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

The directory structure should look like this
```
$REPO_ROOT
    -> data
        -> train
            -> images
                -> 0
                    *.png
                -> 1
                ...
                -> 9
                    *.png
        -> test
            -> images
                -> 0
                    *.png
                ...
    -> dataset
    -> tools
        
```
For setting up the celeb dataset:
* Simple Download the images from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) -> Downloads -> Align&Cropped Images
* Download the `img_align_celeba.zip` file from drive link
* Extract it under the root directory of the repo
* $REPO_ROOT -> img_align_celeba/*.jpg files
        

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/DCGAN-Pytorch.git```
* ```cd DCGAN-Pytorch```
* ```pip install -r requirements.txt```
* ```python -m tools.train --config config/mnist.yaml``` for training and saving inference samples on mnist

## Configuration
* ```config/mnist.yaml``` -  For mnist
* ```config/mnist_colored.yaml``` -  For mnist colored images
* ```config/celeb.yaml``` -  For training on celeb dataset


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created 
* Best Model checkpoints(discriminator and generator) in ```task_name``` directory
* Generated samples saved in ```task_name/samples``` 





