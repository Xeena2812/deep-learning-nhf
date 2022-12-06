import image_preprocessing as pre_proc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import cv2



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


for f in range(0,10):
    img=np.zeros((256,256))
    k=f*16
    for i in range(0,4):
        for j in range(0,4):
            img[i*63:i*63+64,j*63:j*63+64]=generated_image[k,:,:,0]
            k+=1
    ig, axs=plt.subplots(1,3)
    plt.subplot(1,3,1)
    axs[0].set_title('Downscaled')
    axs[1].set_title('Rescaled')
    axs[2].set_title('Original')
    plt.imshow(X_test[f], cmap='gray',label='asd')
    plt.subplot(1,3,2)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(Y_test[f], cmap='gray')
    plt.savefig('full_image/evaluate{}.png'.format(f))
    plt.close()

psnr=[]
psnr2=[]
for i in range(0,160):
    cv2.imwrite("image.png",np.asarray(generated_image[i]*255))
    image=cv2.imread("image.png")

    cv2.imwrite("image.png",np.asarray(y[i]*128))
    imagey=cv2.imread("image.png")
    
    cv2.imwrite("image.png",np.asarray(x[i].reshape((64,64))*128))
    imagex=cv2.imread("image.png")
    
    psnr.append(tf.image.psnr(imagey,image,128))
    psnr2.append(tf.image.psnr(imagex,image,128))
    
print("Peak signal-to-noise ratio of rescaled and original:")
print("AVG:")
print(np.asarray(psnr).mean())
print("MIN:")
print(np.asarray(psnr).min())
print("MAX:")
print(np.asarray(psnr).max())

print("Peak signal-to-noise ratio of downscaled and original:")
print("AVG:")
print(np.asarray(psnr2).mean())
print("MIN:")
print(np.asarray(psnr2).min())
print("MAX:")
print(np.asarray(psnr2).max())

