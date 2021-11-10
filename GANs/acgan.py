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


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, n_classes):
        super().__init__()
        """
        latent_dim: length of the latent space
        img_shape: shape of the image
        n_classes: number of classes 
        """
        self.img_shape = img_shape

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        z = torch.mul(self.label_emb(labels), z)
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, n_classes):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        labels = self.aux_layer(out)

        return validity, labels


class ACGAN(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 n_classes: int = 10, **kwargs):
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
        # number of classes
        self.n_classes = n_classes

        # networks
        # shape of the image (padded to 32 * 32)
        img_shape = (1, 32, 32)
        # generator of Gan
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape, n_classes=n_classes).cuda()
        # discriminator of Gan
        self.discriminator = Discriminator(img_shape=img_shape, n_classes=n_classes).cuda()
        # random noise for generation
        self.validation_z = torch.randn(5, self.latent_dim).cuda()

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
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, drop_last=True)

    def forward(self, z, labels):
        """
        forward pass for generator
        """
        return self.generator(z, labels)

    def adversarial_loss(self, y_hat, y):
        """
        a cross entropy loss for the discriminator to distinguish real or fake
        """
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def auxiliary_loss(self, y_hat, y):
        """
        auxiliary loss to help the discriminator classify the class
        """
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        batch: one batch of real images
        batch_idx: index of the batch, now is all fake
        optimzier_idx: we have two optimizers, 0 for the generator, 1 for the discriminator
        """
        # get the images and labels
        imgs, labels = batch
        imgs = imgs.cuda()
        labels = labels.cuda()

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)
        # assign random labels for image generation
        gen_labels = torch.LongTensor(np.random.randint(0, self.n_classes, self.batch_size)).cuda()

        # train generator using optimizer-0
        if optimizer_idx == 0:
            # generate fake images from random noise
            self.generated_imgs = self(z, gen_labels)

            # log sampled images into tensorboard
            sample_imgs = self.generated_imgs[:6]
            #grid = torchvision.utils.make_grid(sample_imgs)
            #self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss:we fake the discriminator and says the fake images are valid(1)
            validity, pred_label = self.discriminator(self(z, gen_labels))
            g_loss = 0.5 * self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels)

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
            gen_imgs = self(z, gen_labels)

            validity, pred_label = self.discriminator(imgs)
            real_loss = 0.5*self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, labels)

            # fake images labeled as fake (0)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_validity, fake_pred_label = self.discriminator(gen_imgs)
            fake_loss = 0.5*self.adversarial_loss(fake_validity, fake) + self.auxiliary_loss(fake_pred_label, gen_labels)

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
        z = self.validation_z.to(self.device)
        labels = torch.LongTensor([1, 2, 3, 4, 5]).cuda()

        # log the generated images during validation
        sample_imgs = self(z, labels)
        grid = torchvision.utils.make_grid(sample_imgs)

        if (self.current_epoch % 20) == 0:
            print(self.current_epoch)
            tensorboard = self.logger.experiment
            tensorboard.add_image('generated_images_' + str(self.current_epoch), grid, self.current_epoch)

        #self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main(args: Namespace) -> None:
    # setup the model
    model = ACGAN(**vars(args))

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
# python acgan.py
# VISUALIZATION
# tensorboard --logdir lightning_logs/
