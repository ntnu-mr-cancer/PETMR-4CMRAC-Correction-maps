import argparse
import os
from util import util
import torch
import models
import data
import sys

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc for aligned dataset and train/A, train/B, val/A, val/B, test/A, test/B for np_array_aligned)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [pix2pix | cnn | test]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal | lecun_normal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--padding_type', type=str, default='reflect', help='specify padding in generator architecture [zero | reflect | replicate ]')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='np_array_aligned', help='chooses how datasets are loaded. [aligned | np_array_aligned]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # options to split and label the color channels in the display
        parser.add_argument('--display_split_A_color_channels', action='store_true', help='displays the RGB channels of the A image as separate images')
        parser.add_argument('--display_name_A_R', type=str, default = 'A_R', help = 'display name for the R channel of the real_A image. Only if display_split_A_color_cahnnels.')
        parser.add_argument('--display_name_A_G', type=str, default = 'A_G', help = 'display name for the G channel of the real_A image. Only if display_split_A_color_cahnnels.')
        parser.add_argument('--display_name_A_B', type=str, default = 'A_B', help = 'display name for the B channel of the real_A image. Only if display_split_A_color_cahnnels.')
        # Scale an normalize input images
        # Shift and scale A image cumputed as (input-A_bias)/A_range
        parser.add_argument('--no_A_scaling', action='store_true', help = 'Normalize the A image using the parameters specified in --A_bias and --A_range. On by default.')
        parser.add_argument('--A_bias', type=float, default = 0.5, help = 'Bias used in normalization of the A images. The images will be subtracted by this number then divided B_range during normalization. Will only work for the np_array_aligned dataset type.')
        parser.add_argument('--A_range', type=float, default = 0.5, help = 'Range used in normalization of the A images. The images will be subtracted by the bias and divided by the range during normalization. Will only work for the np_array_aligned dataset type.')
        # B
        # Shift and scale B image cumputed as (input-B_bias)/B_range
        parser.add_argument('--scale_B', action='store_true', help = 'Normalize the B image using the parameters specified in --B_bias and --B_range. Off by default.')
        parser.add_argument('--B_bias', type=float, default = 0.5, help = 'Bias used in normalization of the B images. The images will be subtracted by this number then divided B_range during normalization. Will only work for the np_array_aligned dataset type.')
        parser.add_argument('--B_range', type=float, default = 0.5, help = 'Range used in normalization of the B images. The images will be subtracted by the bias and divided by the range during normalization. Will only work for the np_array_aligned dataset type.')
        parser.add_argument('--resnet_activation_function', type=str, default='relu', help='Activation function for ResNet-based generators [relu | selu]')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if os.path.exists(expr_dir) and self.isTrain:
            if not opt.force_overwrite:
                abort = input(f"{expr_dir} already exists do you want to continue? (Y)/N\t")
                while abort not in ['Y', 'N','']:
                    print('Invalid input')
                    abort = input(f"{expr_dir} already do you want to continue? (Y)/N")
                if abort == 'N':
                    sys.exit()

        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def model_specific_options_warning(self, opt):
        # opt = self.gather_options()
        model_specific_options = {
            'pix2pix' : [
                'gan_mode',
                'netD',
                'ndf',
                'n_layers_D',
                'lambda_L1'
            ],
            'cnn' : [
                'cnn_loss'
            ]
        }
            
        message = "The following model specific options will be ignored and removed:\n"
        message += '----------------- Ignored options ---------------\n'
        model = opt.model
        if 'unet' in opt.netG:
            model_specific_options[model] = 'resnet_activation_function'
                
        for model in [model for model in model_specific_options.keys() if model != opt.model]:
            for key in [key for key in model_specific_options[model] if key in opt.__dict__.keys()]:
                print(key, opt.__dict__[key])
                message += '{:>25}: {:<30}\n'.format(str(key), str(opt.__dict__[key]))
                del opt.__dict__[key]
        message += '--------------------- End -----------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.model_specific_options_warning(opt)

        self.print_options(opt)
        
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

# def warn_incompatible_arguments(self):
    