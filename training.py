import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import time
import tensorflow as tf
import keras
from keras.models import Model
import image_preprocessing as pre_proc

def make_generator_model():
    y=[]

    input=layers.Input(shape=(64*64,))
    x=layers.Reshape((64,64,1,))(input)
    x=layers.Conv2D(32,(1,1),strides=1,padding='same')(x)

    

    y.append(x)
    x=DenseBlock(x, 4)
    y.append(x)
    x=layers.Concatenate(axis=3)(y)
    
    x=layers.Conv2D(32,(1,1),strides=1,padding='same')(x)
    x=DenseBlock(x, 4)
    y.append(x)
    x=layers.Concatenate(axis=3)(y)
    
     
    
    x=layers.Conv2D(32,(1,1),strides=1,padding='same')(x)
    x=DenseBlock(x, 4)
    y.append(x)
    x=layers.Concatenate(axis=3)(y)
    x=layers.Conv2D(32,(1,1),strides=1,padding='same')(x)
    x=DenseBlock(x, 4)
    y.append(x)
    x=layers.Concatenate(axis=3)(y)
    
    
    
    x=layers.Conv2D(1,(1,1),strides=1,padding='same')(x)
    
    #x=layers.Reshape((64,64,1))(x)

    asd = Model(inputs=input, outputs=x)
    return asd

def DenseBlock(x, size):
    skip=[]
    skip.append(x)
    for i in range(0,size) :
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
        
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
        
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
        
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
    
    return x

def ConvBlock(x,kernel_size) :
    x=layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x=layers.Conv2D(16,kernel_size,strides=1,padding='same')(x)
    x=layers.Conv2D(16,(1,1),strides=1,padding='same')(x)
    #x=layers.Dropout(0.3)(x)
  
    
    
    return x

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same',
                                     input_shape=[64, 64, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=2, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=1, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=2, padding='same'))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

tf.random.set_seed(5)
def train_generator(generator, X, Y, X_valid, Y_valid, learning_rate, batch_size, epochs):
  generator.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
  generator.fit(X,Y,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(X_valid,Y_valid),shuffle= True)

dataset_size=200
train_split=0.7
valid_split=0.2
test_split=0.1

startx=32
starty=startx+64

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = pre_proc.load_and_transform_data(train_split, valid_split,test_split , dataset_size)

EPOCHS = 1
noise_dim = 64*64
num_examples_to_generate = 1000
BUFFER_SIZE = 600000
BATCH_SIZE = 2

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])



train_generator(generator,X_train,Y_train,X_valid,Y_valid,0.0001,2,50)

generator.save('model')

