"""
modified from https://github.com/nocotan/pytorch-lightning-gans for teaching purpose
"""
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        """
        latent_dim: length of the latent space
        img_shape: shape of the image
        """
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        """
        img_shape: shape of the image
        """

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        # saving hyper-parameters for display purpose in tensorboard
        self.save_hyperparameters()
        # dimension of the latent space
        self.latent_dim = latent_dim
        # learning rate
        self.lr = lr
        # parameters of adams optimizer, b1 and b2
        self.b1 = b1
        self.b2 = b2
        # batch size
        self.batch_size = batch_size

        # networks
        # shape of the image
        mnist_shape = (1, 28, 28)
        # generator of Gan
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=mnist_shape)
        # discriminator of Gan
        self.discriminator = Discriminator(img_shape=mnist_shape)
        # random noise for generation
        self.validation_z = torch.randn(8, self.latent_dim)

    def configure_optimizers(self):
        """
        configure the optimizers
        """
        # learning rate lr
        lr = self.lr
        # parameters of Adams optimizer, b1 & b2
        b1 = self.b1
        b2 = self.b2
        # optimizer for generator, opt_g
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        # optimizer for the discriminator, opt_d
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        # optimizers will be called following the list, 0 for opt_g and 1 for opt_d
        return [opt_g, opt_d], []

    def train_dataloader(self):
        """
        we only have training data from real images
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def forward(self, z):
        """
        forward pass for generator
        """
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        """
        a cross entropy loss for the discriminator to distinguish real or fake
        """
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        batch: one batch of real images
        batch_idx: index of the batch, now is all fake
        optimzier_idx: we have two optimizers, 0 for the generator, 1 for the discriminator
        """
        # get the images
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        # train generator using optimizer-0
        if optimizer_idx == 0:
            # generate fake images from random noise
            self.generated_imgs = self(z)

            # log sampled images into tensorboard
            sample_imgs = self.generated_imgs[:6]
            #grid = torchvision.utils.make_grid(sample_imgs)
            #self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss:we fake the discriminator and says the fake images are valid(1)
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)

            # return the outputs
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator using optimizer-1
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # train the discriminator in the honest way: real images labeled as valid (1)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # fake images labeled as fake (0)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of real_loss and fake_loss
            d_loss = (real_loss + fake_loss) / 2

            # return the outputs
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def on_epoch_end(self):
        """
        during validation, we create additional noise z to generate totally new images
        """
        z = self.validation_z.to(self.device)

        # log the generated images during validation
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)

        if (self.current_epoch % 20) == 0:
            print(self.current_epoch)
            tensorboard = self.logger.experiment
            tensorboard.add_image('generated_images_' + str(self.current_epoch), grid, self.current_epoch)

        #self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main(args: Namespace) -> None:
    # setup the model
    model = GAN(**vars(args))

    # set-up the trainer
    trainer = Trainer(gpus=args.gpus)

    # training in one line
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)


# USAGE
# RUN
# python gan.py
# VISUALIZATION
# tensorboard --logdir lightning_logs/
