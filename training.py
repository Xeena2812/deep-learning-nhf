
from tensorflow.keras import layers

import tensorflow as tf
import keras
from keras.models import Model
import image_preprocessing as pre_proc


#the base model
#has one dense block that contains 8 convolutional blocks
def make_generator_model(   num_of_dense=2, #number of dense blocks
                            num_of_conv=2): #number of convolutional blocks in a dense block
    y=[]
    input=layers.Input(shape=(64*64,))
    x=layers.Reshape((64,64,1,))(input)
    x=layers.Conv2D(32,(1,1),strides=1,padding='same')(x)
    y.append(x)
    for i in range(0 ,num_of_dense):
        x=DenseBlock(x, num_of_conv)
        y.append(x)
        x=layers.Concatenate(axis=3)(y)
        x=layers.Conv2D(1,(1,1),strides=1,padding='same')(x)

    asd = Model(inputs=input, outputs=x)
    return asd

def DenseBlock( x, #Model to add to
                size): #size of dense block  ---  number of convolutional blocks = size*2
    skip=[]
    skip.append(x)
    for i in range(0,size) :
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
        x=ConvBlock(x,(3,3))
        skip.append(x)
        x=layers.Concatenate(axis=3)(skip)
    return x

def ConvBlock(  x, #Model to add to
                kernel_size) : #size of the convolution
    x=layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x=layers.Conv2D(16,kernel_size,strides=1,padding='same')(x)
    x=layers.Conv2D(16,(1,1),strides=1,padding='same')(x)
    return x
    

#Compiles the model and runs the training of the model
def train_generator(    generator, #the model
                        X, #
                        Y, #the training dataset
                        X_valid, #
                        Y_valid, #the validation dataset
                        learning_rate, #learning rate for Adam optimizer
                        batch_size, 
                        epochs):
    generator.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
    generator.fit(X,Y,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(X_valid,Y_valid),shuffle= True)




#loads the dataset, runs the training and saves the model

def train(  num_of_dense=2, #number of dense blocks
            num_of_conv=2, #number of convolutional blocks in a dense block
            dataset_size=200, #Number of individual mri-s to load
            learning_rate=0.0001, #Learning rate for Adam optimizer
            batch_size=2,
            epochs=1):

    train_split=0.7
    valid_split=0.2
    test_split=0.1


    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = pre_proc.load_and_transform_data(train_split, valid_split,test_split , dataset_size)
    generator = make_generator_model(num_of_dense,num_of_conv)
    tf.random.set_seed(5)


    train_generator(generator,X_train,Y_train,X_valid,Y_valid,learning_rate,batch_size,epochs)
    generator.save('model')

