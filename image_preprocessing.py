import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import gdown
import zipfile
import gzip
import shutil
import cv2
from sklearn import preprocessing
import math
from numpy.fft import fft, ifft


"""
    We use the Automated Cardiac Diagnosis Challenge (ACDC) database. https://acdc.creatis.insa-lyon.fr/#challenges
    After signing up the training and validation datasets are available to download.
    The databases contain a total of 150 MRI scans from different patients, some healthy and some with different conditions. 
    For ease of access and persistence we uploaded the datasets to google drive, from where the script downloads them.
"""
def download_db():
    training_url = 'https://drive.google.com/u/0/uc?id=1FTUNw1gIYjIphJBfF9p-cAEa2fZVz76G&export=download'
    testing_url = 'https://drive.google.com/u/0/uc?id=1BTPWTKHxNDXgOr1jM3S6iYM0clGjHld4&export=download'

    # Download the zips from the given urls
    gdown.download(training_url, 'training.zip', quiet=False, verify=False)
    gdown.download(testing_url, 'testing.zip', quiet=False, verify=False)


"""
    Unzip the training and testing datasets and the scans in them into the train_valid/patients directory. 
    A couple of scans (38, 57, 85, 89, 100, 147) are larger than 256 pixels, so we can not use them.
"""
def extract_files():
    os.mkdir('train_valid')

    with zipfile.ZipFile('training.zip', 'r') as zip_ref:
        zip_ref.extractall('train_valid')

    with zipfile.ZipFile('testing.zip', 'r') as zip_ref:
        zip_ref.extractall('train_valid')

    os.mkdir('./train_valid/patients')
    os.remove('./train_valid/training/patient001.Info.cfg')
    shutil.rmtree('./train_valid/testing/testing')

    for subdir in os.listdir('./train_valid/training'):
        with gzip.open(f'./train_valid/training/{subdir}/{subdir}_4d.nii.gz', 'rb') as f_in:
            with open(f'./train_valid/patients/{subdir}.nii', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    for subdir in os.listdir('./train_valid/testing'):
        with gzip.open(f'./train_valid/testing/{subdir}/{subdir}_4d.nii.gz', 'rb') as f_in:
            with open(f'./train_valid/patients/{subdir}.nii', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    shutil.rmtree('./train_valid/training')
    shutil.rmtree('./train_valid/testing')
    #os.remove('./training.zip')
    #os.remove('./testing.zip')
    os.remove('./train_valid/patients/patient038.nii')
    os.remove('./train_valid/patients/patient057.nii')
    os.remove('./train_valid/patients/patient085.nii')
    os.remove('./train_valid/patients/patient089.nii')
    os.remove('./train_valid/patients/patient100.nii')
    os.remove('./train_valid/patients/patient147.nii')

"""
    The scans come as 4D .nii file, which are 3D MRI scans over some time.
    We unroll the scans into 2D images and add them to an original_images folder. The images will be the reference for the super resolution images.
    We store the unrolled images in the /train_valid/original_images directory.
    Note that this operation will take a long time as there will be ~36000 images as the result. This also takes up a considerable amount of disk space ~2GB.
    Not all scans come as the same size so we pad them with '0's to make them 256x256 pixels. The few scans larger than 256 pixels have to be cleared out before this.
"""
def unroll_scale_images():
    os.mkdir('./train_valid/original_images')
    os.mkdir('./train_valid/downscaled_images')
    
    for file in os.listdir('./train_valid/patients'):
        img=nib.load(f'./train_valid/patients/{file}')
        img_data = img.get_fdata()

        for i in range(img.shape[2]):
            for j in range(img.shape[3]):
            # Right now only takes first 3 images in the Z direction, as a way to reduce data and later to not use unnecessary images
            # for j in range(3):
                expanded_image = img_data[:, :, i, j]

                ft=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(expanded_image)))
                ft[:75,:]=0
                ft[:,:75]=0
                ft[150:,:]=0
                ft[:,150:]=0

                inverse_ft=np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(ft))).real
                
                x_diff = (256 - expanded_image.shape[1]) // 2
                y_diff = (256 - expanded_image.shape[0]) // 2

                expanded_image = np.pad(expanded_image, ((y_diff, y_diff), (x_diff, x_diff)), 'constant')
                downscaled_image = np.pad(inverse_ft, ((y_diff, y_diff), (x_diff, x_diff)), 'constant')
                
                cv2.imwrite(f'./train_valid/original_images/{file[:-4]}_{i}_{j}.png', expanded_image)
                cv2.imwrite(f'./train_valid/downscaled_images/{file[:-4]}_{i}_{j}.png', downscaled_image)
        
        print(file, 'done')


"""
    This shows an example of how an original image looks like versus how its scaled down version looks like.
"""
def show_example_images():
    full_scale = plt.imread('./train_valid/original_images/patient001_0_0.png')
    scaled_down = plt.imread('./train_valid/downscaled_images/patient001_0_0.png')

    plt.subplot(1, 2, 1)
    plt.imshow(full_scale, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(scaled_down, cmap='gray')


"""
    Creating training, validation and test datasets. After creation the sets are
"""
def load_and_transform_data(train_split, valid_split, test_split, num_images=100):
    # goes to other file
    # #np.random.seed(2812)

    dataset_path = './train_valid/downscaled_images/'

    X = []
    Y = []

    # Use reduced databse with num_images
    files = os.listdir(dataset_path)
    for i in range(num_images):
        X.append(plt.imread(dataset_path+files[i]).flatten())
        Y.append(plt.imread('./train_valid/original_images/'+files[i]).flatten())

    # Use full database
    # X = [plt.imread(dataset_path+image) for image in os.listdir(dataset_path)]
    # Y = [plt.imread('./train_valid/original_images/'+image) for image in os.listdir('./train_valid/original_images/')]

    dataset_size = len(X)

    train_size = math.ceil(train_split * dataset_size)
    validate_size = int(train_size + valid_split * dataset_size)


    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_valid = X[train_size:validate_size]
    Y_valid = Y[train_size:validate_size]
    X_test = X[validate_size:]
    Y_test  = Y[validate_size:]

    scaler = preprocessing.StandardScaler()

    #X_train = scaler.fit_transform(X_train)
    #X_valid = scaler.fit_transform(X_valid)
    #X_test = scaler.fit_transform(X_test)

    X_train = [i.reshape(256, 256) for i in X_train]
    X_valid = [i.reshape(256, 256) for i in X_valid]
    X_test = [i.reshape(256, 256) for i in X_test]
    
    randperm = np.random.permutation(len(X_train))
    X_train,Y_train = np.array(X_train)[randperm.astype(int)], np.array(Y_train)[randperm]


    Y_valid=np.asarray(Y_valid).reshape((int(valid_split*num_images),256,256))
    Y_train=np.asarray(Y_train).reshape((int(train_split*num_images),256,256))



    y=[]
    for elem in Y_train:
        for i in range(0,4):
            for j in range(0,4):
                y.append(elem[i*64:i*64+64,j*64:j*64+64])
    y=np.asarray(y).reshape((int(train_split*num_images)*16,64,64))
    

    x=[]
    for elem in X_train:
        for i in range(0,4):
            for j in range(0,4):
                x.append(elem[i*64:i*64+64,j*64:j*64+64])
    x=np.asarray(x).reshape((int(train_split*num_images)*16,4096))

    z=[]
    for elem in Y_valid:
        for i in range(0,4):
            for j in range(0,4):
                z.append(elem[i*64:i*64+64,j*64:j*64+64])
    z=np.asarray(z).reshape((int(valid_split*num_images*16),64,64))
    

    zs=[]
    for elem in X_valid:
        for i in range(0,4):
            for j in range(0,4):
                zs.append(elem[i*64:i*64+64,j*64:j*64+64])
    zs=np.asarray(zs).reshape((int(valid_split*num_images*16),64*64))

    plt.imshow(X_test[0])

    return x, y, zs, z, X_test, Y_test
