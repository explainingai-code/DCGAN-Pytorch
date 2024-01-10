# DCGAN-Pytorch


## Data preparation

we will use the [Celeb-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)_ which can be downloaded at the linked site, or in [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)_. The dataset will download as a file named ``img_align_celeba.zip``.
Once downloaded, extract the zip inside the ``/data`` folder. he resulting directory structure should be:


::
```
/data/img_align_celeba/
    -> img_align_celeba  
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
            ...
```

# Quickstart
* ```git clone https://github.com/explainingai-code/DCGAN-Pytorch.git``
* ```cd DCGAN-Pytorch```
* ```pip install -r requirements.txt```
* ```python -u tools/train_dcgan.py``` for training and saving inference samples

## Output 
Outputs will be saved every 50 steps in `samples` directory .

During training of GAN the following output will be saved 
* Latest Model checkpoints for generator and discriminator  in ```$REPO_ROOT``` directory

During inference every 50 steps the following output will be saved
* Sampled image grid for in ```samples/*.png``` 