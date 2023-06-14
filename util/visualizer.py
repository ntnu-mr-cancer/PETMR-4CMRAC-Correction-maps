import os
import ntpath
from . import util, utilBitDepth

def save_images(opt, webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        opt                      -- stores all the experiment flags; need to be subclass of BaseOptions
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    # Grabbing the correct tensor2im function according to the dataset.
    if opt.dataset_mode == 'np_array_aligned':
        tensor2im = utilBitDepth.tensor2im
    else: 
        tensor2im = util.tensor2im

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if opt.saveNpyImages:
            if opt.saveOnlyFakeBImage:
                if label == 'fake_B':
                    utilBitDepth.saveVisualsAsNpy(im_data, os.path.join(os.path.dirname(save_path), *['..', 'npyImages', os.path.basename(save_path)]).replace('.png',''))
            else:
                utilBitDepth.saveVisualsAsNpy(im_data, os.path.join(os.path.dirname(save_path), *['..', 'npyImages', os.path.basename(save_path)]).replace('.png',''))
    webpage.add_images(ims, txts, links, width=width)