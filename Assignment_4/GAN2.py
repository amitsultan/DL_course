import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import activations, regularizers
from tensorflow.keras.optimizers import Adam


class GAN():

    def __init__(self, batch_size, lr, noise_dim, data_dim):
        self.batch_size = batch_size
        self.lr = lr
        self.noise_dim = noise_dim
        self.data_dim = data_dim

        self.generator = Generator(). \
            build_model(noise_dim=self.noise_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(). \
            build_model(input_shape=self.data_dim)

        optimizer = Adam(lr, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  # wasteful to shuffle every time
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        x = train.loc[train_ix[start_i: stop_i]].values
        return np.reshape(x, (batch_size, -1))

    def train(self, data, cache_prefix, epochs, sample_interval):
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        np.zeros((self.batch_size, 1))
        history = {'D_loss': [],
                   'D_acc': [],
                   'G_loss': []}
        for epoch in range(epochs):
            for batch_idx in range(int(np.ceil(len(data)/self.batch_size))):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                batch_data = self.get_data_batch(data, self.batch_size)
                noise = tf.random.normal((self.batch_size, self.noise_dim))

                # Generate a batch of new images
                gen_data = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(batch_data, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = tf.random.normal((self.batch_size, self.noise_dim))
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
            
            if epoch % sample_interval == 0:
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            history['D_loss'].append(d_loss[0])
            history['D_acc'].append(100 * d_loss[1])
            history['G_loss'].append(g_loss)
            # If at save interval => save generated events
            if epoch % sample_interval == 0:
                # Test here data generation step
                # save model checkpoints
                model_checkpoint_base_name = 'model/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))
        return history

    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return

    def load(self, path):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator


class Generator():
        
    def build_model(self, noise_dim, data_dim):
        input = Input(shape=noise_dim)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(input)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(data_dim, activation='tanh', kernel_initializer='glorot_normal')(x)
        return Model(inputs=input, outputs=x)


class Discriminator():

    def build_model(self, input_shape):
        input = Input(shape=input_shape)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation='sigmoid')(x)
            # opt = tensorflow.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        return Model(inputs=input, outputs=x)
    
    