import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam


from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

import tensorflow as tf
class GAN_RF():

    def __init__(self, gan_args, clf):
        [self.batch_size, lr, self.noise_dim,
         self.data_dim, layers_dim, self.C] = gan_args
        self.clf = clf
        self.generator = Generator_RF(self.batch_size, self.C). \
            build_model(input_shape=(self.noise_dim,), dim=layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator_RF(self.batch_size, self.C). \
            build_model(input_shape=(self.data_dim,), dim=layers_dim)

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        c = Input(shape=(1,))
        record = self.generator([z, c])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        y = Input(shape=(1,))
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator([record, y, c])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, y,c], validity)
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

    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments

        data_cols = data.columns

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        # C = np.full((self.batch_size, 1), 0.8)
        history = {'D_loss': [],
                   'D_acc': [],
                   'G_loss': []}
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            C = np.random.uniform(0, 1, self.batch_size)
            batch_data = self.get_data_batch(data, self.batch_size)
            noise = tf.random.normal((self.batch_size, self.noise_dim))

            # Generate a batch of new images
            gen_data = self.generator.predict([noise, C])
            batch_x = batch_data[:,:-1]
            bb_y = self.clf.predict_proba(batch_x)
            # Train the discriminator
            bb_gen_data_y = self.clf.predict_proba(gen_data[:,:-1])
            d_loss_real = self.discriminator.train_on_batch([batch_data, bb_y[:,0], C], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_data, bb_gen_data_y[:,0], C], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([noise, valid, C], valid)
            
            if epoch % 100 == 0:
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

class Generator_RF():
    def __init__(self, batch_size, C):
        self.batch_size = batch_size
        self.C = C

    def build_model(self, input_shape, dim, data_dim):
        input_Z = Input(shape=input_shape, batch_size=self.batch_size)
        input_C = Input(shape=1, batch_size=self.batch_size)
        x = Concatenate(axis=1)([input_Z, input_C])
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=[input_Z, input_C], outputs=x)
    
class Discriminator_RF():
    def __init__(self, batch_size, C):
        self.batch_size = batch_size
        self.C = C

    def build_model(self, input_shape, dim):
        input_sample = Input(shape=input_shape, batch_size=self.batch_size)
        input_Y = Input(shape=1, batch_size=self.batch_size)
        input_C = Input(shape=1, batch_size=self.batch_size)
        x = Concatenate(axis=1)([input_sample, input_Y, input_C])
        x = Dense(dim * 4, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_sample, input_Y, input_C], outputs=x)