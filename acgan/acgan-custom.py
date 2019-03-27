#########################################################
#########################################################

# Author: Chen-Yi Lu
# Department: NTU BIME BBLAB
# LAST MODIFIED: 3/20/2019

# REFERENCE: https://arxiv.org/pdf/1606.03498.pdf
#            https://github.com/eriklindernoren/Keras-GAN

#########################################################
#########################################################


from __future__ import print_function, division

import tensorflow as tf
import keras 

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from scipy import ndimage
from keras.preprocessing.image import array_to_img, img_to_array

import matplotlib.pyplot as plt

import sys

import numpy as np

from PIL import Image

import os, sys



print(tf.__version__)
print(keras.__version__)



class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.latent_dim = 100

        optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        #self.discriminator = multi_gpu_model(self.discriminator, gpus=2)
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        #model.add(Dense(1024))
        model.add(Dense(768 , activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 12)))
        model.add(Conv2DTranspose(384, kernel_size=5, strides=(2,2), activation='relu', padding="same"))
        model.add(BatchNormalization(momentum=0.8))               
        model.add(Conv2DTranspose(256, kernel_size=5, strides=(2,2), activation='relu', padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(192, kernel_size=5, strides=(2,2), activation='relu', padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(3, kernel_size=5, strides=(2,2), activation="tanh", padding="same"))    
        

        
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Flatten())

        #model = multi_gpu_model(model, gpus=2)
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        train_dir = 'D:/Steven_Data/TRAINING_DATA/ori_training_data_7-12'

        class_name = os.listdir(train_dir)
        class_num = 0
        train_files = []
        train_labels = []
        for classes in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, classes)
            class_num += 1
            print(classes+"\n")
            for file in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, file)):
                    train_files.append(os.path.join(class_dir, file))
                    train_labels.append(class_num)
                    
                        
        # train_files = np.array(train_files)
        # print(train_files.shape)
        train_labels = np.array(train_labels)
        train_labels = train_labels.reshape(-1,1)
        #print(train_labels.shape)
        
        image_rows = 128
        image_cols = 128
        channels = 3
        dataset = np.ndarray(shape=(len(train_files), image_rows, image_cols, channels),
                     dtype=np.float32)
        i = 0
        for _file in train_files:
            img = load_img(_file)  # this is a PIL image
            img = img.resize((image_rows, image_cols))
            # Convert to Numpy Array
            x = img_to_array(img)  
            dataset[i] = x
            i += 1
            
        # Rescale -1 to 1
        print("%d images to array" % len(dataset))
        dataset = (dataset.astype(np.float32) - 127.5) / 127.5
        print(dataset.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            imgs = dataset[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])
            #print(gen_imgs.shape)
            # Image labels. 0-6 if image is valid or 7 if it is generated (fake)
            img_labels = train_labels[idx]
            fake_labels = self.num_classes * np.ones(img_labels.shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model(epoch)
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, self.num_classes
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/last_6_months/%d.png" % epoch)
        plt.close()

    def save_model(self, epoch):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
        self.generator.save_weights('D:/Steven_DATA/Keras-GAN/acgan/saved_model/ACGAN_SINGLE_LAST6/generator/gen_%d.h5' % epoch)
        self.discriminator.save_weights('D:/Steven_DATA/Keras-GAN/acgan/saved_model/ACGAN_SINGLE_LAST6/discriminator/dis_%d.h5' % epoch)
    def generate_imgs(self, numbers=500):
         for num in range(numbers + 1):
            r, c = 1, 1
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            self.generator.load_weights("saved_model/ACGAN_SINGLE_LAST6/generator/gen_49900.h5")
            gen_imgs = self.generator.predict([noise, np.array([3])])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            img = plt.imshow(gen_imgs[0, :, :, :])
            plt.rcParams["figure.figsize"] = (1.28, 1.28)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            print("generate image: %d" %num)
            
            img.figure.savefig("generated_imgs/test_%d.jpg" % num, bbox_inches=0, pad_inches=0)
            plt.close()

            

if __name__ == '__main__':
    acgan = ACGAN()
    if(sys.argv[1] == 'train'):
        acgan.train(epochs=50000, batch_size=100, sample_interval=100)
    elif(sys.argv[1] == 'gen'):
        acgan.generate_imgs(100)

