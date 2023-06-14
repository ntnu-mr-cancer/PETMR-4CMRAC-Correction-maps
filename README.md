# PETMR-4CMRAC-Correction-maps
The code for the *Pelvic PET/MR attenuation correction in the image space using generative adversarial networks* will be presented here. 

The code is based on this excellent pytorch [pix2pix implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) which again was inspired by [DCGAN](https://github.com/pytorch/examples/tree/main/dcgan).

## Usage
The easiest way to use the code is through the supplied [dockerfile](docker/dockerfile). See the supplied docker [README.md](docker/README.md) for instructions. If visual studio code is used, the supplied devcontainer can be used to automatically set up a development environment inside the docker container. 

### Preparing data
The preprocessing of the data is performed as described in the paper. The data is expected to be stored in the following folder structure

```cmd
datasetName
├── train
│   ├── A
│   └── B
└── val
    ├── A
    └── B
```

The images in A and B are both saved numpy arrays where each saved file contains one image slice. Normally each file in A would be a (256,256,3) matrix and B a (256,256) matrix. 
A working dummy dataset can be created by running [scripts/create_debug_dataset.py](scripts/create_debug_dataset.py) from the project root folder.

### Training and Testing
The commands used for training and testing can be found in the script [scripts/train_test_pix2pix.sh](scripts/train_test_pix2pix.sh). 
The training is monitored using tensorboard.