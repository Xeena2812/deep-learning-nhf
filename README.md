# UAC - Deep Learning NHF

The Goal of this project is to increase the resolution of low resolution heart MRI scans. MRI (Magnetic Resonance Imaging) is one of the primary methods of biomedical imaging. For an MRI machine to capture high resolution images longer exposure time is needed. However longer exposure times expose scans to problems like the subject moving during the scan, resulting in a blurry scan. To minimize this risk short exposure times are used, which result in stable, but low resolution scans, leading to the same problem from a different angle. In this project we attempt to use deep learning to create super resolution images of low resolution MRI scans.

For the project we use the ACDC challenge database (https://acdc.creatis.insa-lyon.fr/#challenges)

We have decided to start by using 2 different apraches. In one we use the whole scans and in the other we only use smaller images containing only the hearts.

The network is a 2D convolutional GAN and we start by training the generator alone. The inputs are the downscaled images and the outputs the upscaled.

The evaluation, once the network has been tested and trained, will be done using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural SIMilarity)

To run the training image_preprocessing.py needs to be beside training.ipynb.

The project is made by:
- Sipos Levente - NLLIEC
- Horváth Bence - ET2EPO

### Files

full_image_preprocessing.ipynb - downloads the scans and unrolls them into 2D images, then saves copies at the desired downscale for different phases of trining.  
small_heat_preprocessing.ipynb - downloads the preprocessed 96 by 96 pixel png-s and and creates a basic training dataset  
training.ipynb - the network and training  
image_preprocessing.py - image preprocessing functions  

### Sources
https://www.tensorflow.org/tutorials/generative/dcgan  
https://acdc.creatis.insa-lyon.fr/description/databases.html  
