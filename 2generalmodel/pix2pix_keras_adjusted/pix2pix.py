from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import argparse
from config import Config

# def parse_args(args, config):
#     """Parses the arguments to the config instance"""

#     config.batch_size = args.batch_size
#     config.num_epochs = args.num_epochs
#     config.train_file_path = args.train_file
#     config.val_file_path = args.val_file
#     config.num_classes = args.num_classes
#     config.use_weights = args.use_weights
#     config.filters = [args.filters*2**i for i in range(args.num_layers)]
#     config.input_size = args.input_size
#     config.mirror = args.mirror
#     config.rotate = args.rotate
#     config.noise = args.noise

#     return config

# def run():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_name", type=str, default="Patches256")
#     parser.add_argument("--task_name", type=str, default="Test1")
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "export"])
#     parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
#     parser.add_argument("--sample_interval", type=int, default=50, help="interval between sampling of images from generators")
#     parser.add_argument("--savemodel_interval", type=int, default=50, help="interval between model checkpoints")
#     parser.add_argument("--input_rows", type=int, default=256, help="rows")
#     parser.add_argument("--input_cols", type=int, default=256, help="cols")
#     parser.add_argument("--input_channels", type=int, default=3, help="channels")
#     a = parser.parse_args()

#     config = Config()
#     config = parse_args(a, config)

#     return config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Patches256")
parser.add_argument("--task_name", type=str, default="Test1")
parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "export"])
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="learning")
parser.add_argument("--beta1", type=float, default=0.5, help="beta1")
parser.add_argument("--sample_interval_batch", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--savemodel_interval_epoch", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--input_rows", type=int, default=256, help="rows")
parser.add_argument("--input_cols", type=int, default=256, help="cols")
parser.add_argument("--input_channels", type=int, default=3, help="channels")
a = parser.parse_args()


class Pix2Pix():
    def __init__(self, a):
        # Input shape
        self.img_rows = a.input_rows
        self.img_cols = a.input_cols
        self.channels = a.input_channels
        self.task_name = a.task_name
        self.epochs = a.max_epochs
        self.lr = a.lr
        self.beta1 = a.beta1
        self.sample_interval = a.sample_interval_batch
        self.savemodel_interval = a.savemodel_interval_epoch
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = a.dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(self.lr, self.beta1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_B = self.generator(img_A)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_B, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_B])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, batch_size=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        model_save_path = 'saved_model/%s' % self.task_name
        os.makedirs(model_save_path, exist_ok=True)

        for epoch in range(self.epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_B = self.generator.predict(imgs_A)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_B, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, self.epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % self.sample_interval == 0:
                    self.sample_images(epoch, batch_i)

                # save model
            if epoch % self.savemodel_interval == 0:

                generator_file_path = os.path.join(model_save_path,'G_%d.hdf5' % epoch)
                discriminator_file_path = os.path.join(model_save_path,'D_%d.hdf5' % epoch)

                self.generator.save(generator_file_path)
                self.discriminator.save(discriminator_file_path)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.task_name, exist_ok=True)
        r, c = 1, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=1, is_testing=True)
        fake_B = self.generator.predict(imgs_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r,c)
        
        cnt = 0
        i, j = 0, 0
        for j in range(c):
            # for j in range(c):
            axs[j].imshow(gen_imgs[cnt])
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.task_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    # config = run()
    gan = Pix2Pix(a)
    gan.train(batch_size=1)
