#!/usr/bin/env python
# coding: utf-8

# * https://github.com/timestocome
# 
# 
# # Kannada MNIST Semi-Supervised GAN
# 
# * Model small set of Kannada MNIST samples of 40,000 samples
# * Holdout and Submission imgs vary from test data in std, mean
# 
# While the GAN slightly outperforms a stardard Supervised FF Conv on Test Data that differs from Training Data it's not by a large amount. Augmenting data and shifting the std and mean to match Test Data is probably a better option
# 
# |n_train samples| GAN validation | GAN holdout | Supervised validation | Supervised holdout |
# | --- | --- | --- | --- | --- |
# | 1000 | 98% | 72% | 97% | 67% |
# | 2000 | 98% | 70% | 97% | 62% |
# | 3000 | 98% | 72% | 98% | 70% |
# | 4000 | 98% | 72% | 97% | 67% |
# | 5000 | 99% | 71% | 98% | 65% |
# | 6000 | 99% | 76% | 98% | 67% |
# | 8000 | 99% | 72% | 98% | 67% |
# | 10000 | 99% | 75% | 98% | 70% |
# | 12000 | 99% | 75% | 99% | 74% | 
# 
# 
# 
# 
# 
# * example code and images from https://www.manning.com/books/gans-in-action
# * dataset https://www.kaggle.com/c/Kannada-MNIST
# 
# ![gan.png](attachment:gan.png)
# 
# 
# * Reducing the Need for Labeled Data in Generative Adversarial Networks https://ai.googleblog.com/2019/03/reducing-need-for-labeled-data-in.html
# 
# * Self-Supervised GANs via Auxiliary Rotation Loss  https://arxiv.org/pdf/1811.11212.pdf
# 
# * Unsupervised Representation Learning by Predicting Image Rotations  https://arxiv.org/abs/1803.07728
# 
# 
# 
# 
# Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script.
# 
# The goal of this Kaggle competition is to use machine learning to correctly label hand-written digits written in the Kannada script. The Kannada dataset format is based on the dataset used in the original MNIST dataset where the objective was the same but the hand-written digits were Arabic numbers.
# 
# 

# In[ ]:


# silence is golden

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


# In[ ]:


# hack to make keras work with 2*** series gpus

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from keras import backend as K

from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Dropout, Flatten, Input, Lambda, Reshape)

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


# In[ ]:


# Kannada MNIST dataset ~ 60,000 samples
# 40,000 labeled train
# 5,000 unlabeled submission
# 10,240 labeled holdout
# find labeled data, split out 100 samples for training


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
holdout = pd.read_csv('../input/Dig-MNIST.csv')


# label, image
#print('train', train.shape)
#print(train.columns)
#print(train.head())


# id, image
#print('test', test.shape)
#print(test.columns)
#print(test.head())


# label, image
#print('holdout', holdout.shape)
#print(holdout.columns)
#print(holdout.head())


# In[ ]:


# split into x image, y label, train -> train/validate

# this is the labeled part of the training set, rest of training data is held aside
num_labeled = 6000


# shuffle train set
train_s = train.sample(frac=1.)
train = train_s[0:30000]
validate = train_s[30000:40000]

print(train.shape, validate.shape)


# split into labeled and unlabeled
train_labeled = train[0:num_labeled]
train_unlabeled = train[num_labeled:len(train)]


y_train_labeled = train_labeled.label.values
x_train_labeled = train_labeled.drop(columns=['label']).values

y_train_unlabeled = train_unlabeled.label.values
x_train_unlabeled = train_unlabeled.drop(columns=['label']).values


y_validate = validate.label.values
x_validate = validate.drop(columns=['label']).values



x_submission = test.drop(columns=['id']).values

y_holdout = holdout.label.values
x_holdout = holdout.drop(columns=['label']).values

print('labeled train', x_train_labeled.shape, y_train_labeled.shape)
print('unlabeled train', x_train_unlabeled.shape, y_train_unlabeled.shape)

print('submission', x_submission.shape)
print('holdout', x_holdout.shape, y_holdout.shape)


# In[ ]:


# train labels 0..9, equal numbers of each
print(y_train_unlabeled.min(), y_train_unlabeled.max())
print(y_train_labeled.min(), y_train_labeled.max())

print(np.unique(y_train_labeled, return_counts=True))
print(np.unique(y_train_unlabeled, return_counts=True))


print(y_holdout.min(), y_holdout.max())
print(np.unique(y_holdout, return_counts=True))


# In[ ]:


# check image data
# original
# 0 255 20.98506292517007 0.0 61.651160617260885
# 0 255 28.725469098772322 0.0 72.93217267514069
# 0 255 18.516213265306124 0.0 57.48610134940895

# center data between -1, 1
def scale_img_data(img):
    return (img - 127.5) / 127.5
    
x_train_labeled = scale_img_data(x_train_labeled)
x_train_unlabeled = scale_img_data(x_train_unlabeled) 
x_validate = scale_img_data(x_validate)
x_holdout = scale_img_data(x_holdout)
x_submission = scale_img_data(x_submission)

                                 
                                 
                                 
                                 
print('notice difference in std and median between training and holdout/validation data')
print(np.min(x_train_labeled), np.max(x_train_labeled), np.mean(x_train_labeled), np.median(x_train_labeled), np.std(x_train_labeled))
print(np.min(x_train_unlabeled), np.max(x_train_unlabeled), np.mean(x_train_unlabeled), np.median(x_train_unlabeled), np.std(x_train_unlabeled))
print(np.min(x_holdout), np.max(x_holdout), np.mean(x_holdout), np.median(x_holdout), np.std(x_holdout))
print(np.min(x_submission), np.max(x_submission), np.mean(x_submission), np.median(x_submission), np.std(x_submission))


# In[ ]:


#  # Expand image dimensions to width x height x channels
def reshape_img_data(x):
    return np.expand_dims(x, axis=3)

x_train_unlabeled = reshape_img_data(x_train_unlabeled)
x_train_labeled = reshape_img_data(x_train_labeled)
x_validate = reshape_img_data(x_validate)
x_holdout = reshape_img_data(x_holdout)
x_submission = reshape_img_data(x_submission)

print(x_train_unlabeled.shape, x_train_labeled.shape, x_validate.shape, x_holdout.shape, x_submission.shape)

# labels
def reshape_labels(y):
    return y.reshape(-1, 1)
    
y_train_unlabeled = reshape_labels(y_train_unlabeled)
y_train_labeled = reshape_labels(y_train_labeled)
y_validate = reshape_labels(y_validate)
y_holdout = reshape_labels(y_holdout)

print(y_train_unlabeled.shape, y_train_labeled.shape, y_validate.shape, y_holdout.shape)


# In[ ]:


img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100

# Number of classes in the dataset
num_classes = 10


# In[ ]:






# random batch of labeled train data
def batch_labeled(batch_size):
        
    x = x_train_labeled
    y = y_train_labeled
        
    # Get a random batch of labeled images and their labels
    idx = np.random.randint(0, num_labeled, batch_size)
    imgs = x[idx]
    labels = y[idx]
    
    imgs = imgs.reshape(batch_size, img_rows, img_cols, 1)
    labels = to_categorical(labels, num_classes=num_classes)

    return imgs, labels

    
# random batch unlabeled train data    
def batch_unlabeled(batch_size):
        
    x = x_train_unlabeled
    y = y_train_unlabeled
        
        
    # Get a random batch of unlabeled images
    idx = np.random.randint(num_labeled, x.shape[0], batch_size)
    imgs = x[idx]
    
    imgs = imgs.reshape(batch_size, img_rows, img_cols, 1)

        
    return imgs


# full training set
def train_set():
    
    imgs = np.concatenate((x_train_labeled, x_train_unlabeled), axis=0)
    y = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)
    
    imgs = imgs.reshape(len(imgs), img_rows, img_cols, 1)
    labels = to_categorical(y, num_classes=num_classes)

    
    return imgs, labels


    

# validation set is a random selection of samples from train set
def test_set():
     
    imgs = x_validate
    y = y_validate
    
    imgs = imgs.reshape(len(imgs), img_rows, img_cols, 1)
    labels = to_categorical(y, num_classes=num_classes)

    return imgs, labels
    
    
    
# unseen set, somewhat different std and mean in images
def holdout_set():
        
    imgs = x_holdout
    y = y_holdout
    
    imgs = imgs.reshape(len(imgs), img_rows, img_cols, 1)
    labels = to_categorical(y, num_classes=num_classes)

    return imgs, labels
    

# unseen set, somewhat different std and mean in images
def submission_set():
        
    imgs = x_submission
    imgs = imgs.reshape(len(imgs), img_rows, img_cols, 1)

    return imgs
        


# # Semi-Supervied GAN
# 
# ![Screenshot%20from%202019-10-05%2014-44-09.png](attachment:Screenshot%20from%202019-10-05%2014-44-09.png)

# ### Generator

# In[ ]:


# model takes in a random noise vector of z_dim and learns to output a digit
# that is good enough that discriminator can't tell it from an MNIST database image



def build_generator(z_dim):

    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))
    

    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    
    # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    
    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    

    return model


# ### Discriminator

# In[ ]:


# takes input image and classifies it from 0...9


def build_discriminator_net(img_shape):

    model = Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    
    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    
    # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))

    
    # Flatten the tensor
    model.add(Flatten())
    model.add(Dense(num_classes))

    return model


# In[ ]:


# discriminator to label class
def build_discriminator_supervised(discriminator_net):

    model = Sequential()
    model.add(discriminator_net)
    model.add(Activation('softmax'))

    return model


# In[ ]:


# discriminator to id real/fake image
def build_discriminator_unsupervised(discriminator_net):

    model = Sequential()
    model.add(discriminator_net)

    # Transform distribution over real classes into a binary real-vs-fake probability
    def predict(x):
        # 1 - 1/Sum(e^x + 1)  --- almost sigmoid function
        prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction

    # 'Real-vs-fake' output neuron defined above
    model.add(Lambda(predict))

    return model


# ## Build the Model

# In[ ]:


def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model


# ### Discriminator

# In[ ]:


# Core Discriminator network:
# These layers are shared during supervised and unsupervised training
discriminator_net = build_discriminator_net(img_shape)


# Build & compile the Discriminator for supervised training
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())


# Build & compile the Discriminator for unsupervised training
discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy', optimizer=Adam())


# ### Generator

# In[ ]:


# Build the Generator
generator = build_generator(z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator_unsupervised.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
# Note that we are using the Discriminator version with unsupervised output
gan = build_gan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())


# ## Training

# In[ ]:


supervised_losses = []
iteration_checkpoints = []



def train(iterations, batch_size, sample_interval):

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    
    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get labeled examples
        imgs, labels = batch_labeled(batch_size)

        # Get unlabeled examples
        imgs_unlabeled = batch_unlabeled(batch_size)

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train on real labeled examples
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)

        # Train on real unlabeled examples
        d_loss_real = discriminator_unsupervised.train_on_batch( imgs_unlabeled, real)

        # Train on fake examples
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))

        if (iteration + 1) % sample_interval == 0:

            # Save Discriminator supervised classification loss to be plotted after training
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print(
                "%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]"
                % (iteration + 1, d_loss_supervised, 100 * accuracy,
                   d_loss_unsupervised, g_loss))


# ## Train the Model and Inspect Output

# Note that the `'Discrepancy between trainable weights and collected trainable'` warning from Keras is expected. It is by design: The Generator's trainable parameters are intentionally held constant during Discriminator training, and vice versa.

# In[ ]:


# Set hyperparameters
n_epochs = 5000
batch_size = 32
sample_interval = 10

# Train the SGAN for the specified number of iterations
train(n_epochs, batch_size, sample_interval)


# In[ ]:


losses = np.array(supervised_losses)

# Plot Discriminator supervised loss
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses, label="Discriminator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Discriminator – Supervised Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()


# ## SGAN Classifier – Training, Test, Holdout Accuracy 
# 

# In[ ]:


# compute training data accuaracy
x, y = train_set()

_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# compute validation data accuaracy
x, y = test_set()

_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# compute holdout data accuracy
x, y = holdout_set()

_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# create Kaggle submission data
# score was not good enough to bother submitting, code here only to show how to pull out ints from one-hot
x = submission_set()

gan_predictions = discriminator_supervised.predict(x)
print('gan preds', gan_predictions.shape)
gan_preds = np.argmax(gan_predictions, axis = 1)

print(gan_preds)


# ---

# # Fully-Supervised Classifier
# 
# * use same number of labeled samples
# * see diff in number of training epoches needed

# In[ ]:


# Fully supervised classifier with the same network architecture as the SGAN Discriminator
mnist_classifier = build_discriminator_supervised(build_discriminator_net(img_shape))
mnist_classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())


# In[ ]:


imgs, labels = batch_labeled(num_labeled)


# Train the classifier
training = mnist_classifier.fit(x=imgs,
                                y=labels,
                                batch_size=batch_size,
                                epochs=n_epochs//100,
                                verbose=1)

losses = training.history['loss']
accuracies = training.history['acc']


# In[ ]:


# Plot classification loss
plt.figure(figsize=(10, 5))
plt.plot(np.array(losses), label="Loss")
plt.title("Classification Loss")
plt.legend()


# In[ ]:


# check training data accuracy

x, y = train_set()
_, accuracy = mnist_classifier.evaluate(x, y)
print("Training Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# check validation data accuracy

x, y = test_set()
_, accuracy = mnist_classifier.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# check holdout data accuracy

x, y = holdout_set()
_, accuracy = mnist_classifier.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# In[ ]:


# create Kaggle submission data

x = submission_set()

clf_predictions = mnist_classifier.predict(x)
print('clf preds', clf_predictions.shape)
clf_preds = np.argmax(clf_predictions, axis = 1)
print(clf_preds)


# ---
