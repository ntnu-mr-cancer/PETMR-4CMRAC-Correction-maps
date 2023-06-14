"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix) and
different datasets (with option '--dataset_mode': e.g., np_array_aligned, ).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/ds_name --name ds_name_pix2pix --model pix2pix --direction AtoB

See options/base_options.py and options/train_options.py for more training options.
"""
import time

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import copy
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os


def print_current_losses(epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    tensorBoardDir = os.path.join(opt.checkpoints_dir, *[opt.name, 'logs'])
    writer = SummaryWriter(tensorBoardDir)

    # Creating validation options and dataset
    if opt.validate:
        valopt = copy.deepcopy(opt)
        valopt.phase = 'val'
        valDataset = create_dataset(valopt)
        valDataset_size = len(valDataset)
        print('The number of validation images = %d' % valDataset_size)
        del valopt # we don't need the validation options anymore.

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            # import pdb; pdb.set_trace()

            if total_iters % opt.display_freq == 0:   # save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                image_path = model.get_image_paths()[0] # We take the first path as this corresponds to the image that is displayed.
                for label, image in model.get_current_visuals().items():
                    writer.add_image(label, make_grid(image, normalize=True), total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                for label, loss in losses.items():
                    writer.add_scalar('loss/' + label, loss, total_iters)

                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            if opt.validate and total_iters % opt.validation_freq == 0:
                val_losses = model.get_validation_losses(valDataset)
                for label, loss in val_losses.items():
                    writer.add_scalar(label, loss, total_iters)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
