import torch
from .base_model import BaseModel
from . import networks


class CNNModel(BaseModel):
    """ This class implements a base CNN model using only the generator architecture.

    The model training requires '--dataset_mode aligned or np_array_aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G']
        # Add rmse loss if specified.
        if self.isTrain:  # For now only defined for training phase
            if opt.rmse:
                self.rmse = True
                self.loss_names.append('RMSE')
            else:
                self.rmse = False
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        padding_type=opt.padding_type, resnet_activation_function=opt.resnet_activation_function)
        if self.isTrain:
            # define loss functions
            if opt.cnn_loss == 'l1':
                self.loss_cnn = torch.nn.L1Loss()
            elif opt.cnn_loss == 'l2':
                self.loss_cnn = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.loss_cnn(self.fake_B, self.real_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # Add RMSE loss if specified
        if self.rmse:
            self.RMSELoss()

    def get_validation_losses(self, valDataset):
        val_loss = torch.empty(len(valDataset))
        for i, val_data in enumerate(valDataset):
            self.set_input(val_data)
            self.test()
            visuals = self.get_current_visuals()
            self.RMSELoss()
            val_loss[i] = self.loss_RMSE
        self.mean_val_loss, self.std_val_loss = torch.mean(
            val_loss), torch.std(val_loss)
        return {'mean_val_loss':  self.mean_val_loss, 'std_val_loss': self.std_val_loss}
