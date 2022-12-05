import image_preprocessing as pre_proc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

os.mkdir("evaluation")

generator=keras.models.load_model('model1')


X_train, Y_train, X_valid, Y_valid, X_test, Y_test = pre_proc.load_and_transform_data(0.7, 0.2,0.1 , 100)


X_test=np.asarray(X_test)

Y_test=np.asarray(Y_test)

Y_test=Y_test.reshape((10,256,256))
X_test=X_test.reshape((10,256,256))

y=[]
for elem in Y_test:
    for i in range(0,4):
        for j in range(0,4):
            y.append(elem[i*64:i*64+64,j*64:j*64+64])

x=[]
for elem in X_test:
    for i in range(0,4):
        for j in range(0,4):
            x.append(elem[i*64:i*64+64,j*64:j*64+64])
x=np.asarray(x).reshape(160,4096)
y=np.asarray(y)

generated_image = generator(x, training=False)


for num in range(0,160):
    plt.imsave('generated.png',generated_image[num,:,:,0])
    fig, axs=plt.subplots(1,3)
    plt.subplot(1,3,1)
    axs[0].set_title('Downscaled')
    axs[1].set_title('Rescaled')
    axs[2].set_title('Original')
    plt.imshow(x[num].reshape(64,64), cmap='gray',label='asd')
    plt.subplot(1,3,2)
    plt.imshow(generated_image[num,:,:,0], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(y[num], cmap='gray')
    plt.savefig('evaluation/evaluate{}.png'.format(num))
    plt.close()



