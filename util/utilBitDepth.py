"""This module mimics the util.py module and contains helper functions primarily
for the visualizer. These functions are used when input and output numpy arrays 
with a bit depth greater than the vanilla 8-bit pngs that are used as input"""
import torch
import numpy as np
from PIL import Image
import os
import numpy as np

def tensor2im(input_image, imtype=np.uint8, scaling = 'FD'):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
        scaling (str, list, tuple) -- specifies the scaling of the output image.
                                      options are FD (full dynamic) and 2 element
                                      list or tuple with high and low value.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    if scaling == 'FD':
        image_numpy = full_dynamic_scale(image_numpy)
    elif isinstance(scaling, (list, tuple)) and len(scaling) == 2:
        image_numpy = (image_numpy - scaling[0])/(scaling[1] - scaling[0])
    else:
        raise ValueError('Unreconized scaling option. Expected list/tuple of length 2 or FD')

    return (image_numpy*255).astype(imtype)

def saveVisualsAsNpy(im_data, image_path, dtype = np.float16):
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    imageNumpy = im_data.data[0].cpu().float().numpy().astype(dtype)
    np.save(image_path, imageNumpy)

def save_image(image_numpy, image_path, aspect_ratio=1.0, scaling = 'FD'):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def full_dynamic_scale(numpy_array):
    arrMin, arrMax = numpy_array.min(), numpy_array.max()
    return (numpy_array - arrMin)/(arrMax - arrMin)