# UAC - Deep Learning NHF

The Goal of this project is to increase the resolution of low resolution heart MRI scans. MRI (Magnetic Resonance Imaging) is one of the primary methods of biomedical imaging. For an MRI machine to capture high resolution images longer exposure time is needed. However longer exposure times expose scans to problems like the subject moving during the scan, resulting in a blurry scan. To minimize this risk short exposure times are used, which result in stable, but low resolution scans, leading to the same problem from a different angle. In this project we attempt to use deep learning to create super resolution images of low resolution MRI scans.

For the project we use the ACDC challenge database (https://acdc.creatis.insa-lyon.fr/#challenges)

We have decided to start by using 2 different apraches. In one we use the whole scans and in the other we only use smaller images containing only the hearts.

The network is a 2D convolutional network based on DenseNet. The inputs are the downscaled images and the outputs the upscaled.

To run the training and the evaluation you will need the following python libraries:
- tensorflow
- nibabel
- matplotlib
- gdown
- zipfile36
- opencv-python
- scikit-learn

After all the libraries have been downloaded you will need to clone this repository, open it and download the dataset.
To do that use the following commands in console:
- cd deep-learning-nhf
- gdown https://drive.google.com/u/0/uc?id=1f3_3hflYwVTNU4qCBJ7YK3wnkUoR6YWi&export=download
- gdown https://drive.google.com/u/0/uc?id=18DXZUo5GyeB0xCejjcxidBWPzqTA6XhH&export=download

Once the previous steps ha been done you can begin the preproceccing, training and evaluation.    
Do these in order:
- run setup.py
- in python
-   import training
-   training.train()
- run evaluate.py

This will run a very short training cycle and evaluate the resulting model.

To run the training that we did write: training.train(4,4,400,0.0001,2,100)

To access the trained model or the resultin evaluation pictures download them from here: https://drive.google.com/drive/folders/1KJ9xrMav3L2WNEiRz3fBjElGAuDRGDg9

To run the evaluations of the pretrained models download them, unzip the chosen model into the repository and run evaluate.py. The previous link also contains the evaluation pictures


The project is made by:
- Sipos Levente - NLLIEC
- Horváth Bence - ET2EPO

### Files

training.py - the creation, training and saving of the model    
image_preprocessing.py - image preprocessing functions  
evaluate.py - evaluating the model and creation of example images   
setup.py - unzipping and unrolling the NII images   
gpu.py - checks weather tensorflow has access to a GPU    


### Sources
https://www.tensorflow.org/tutorials/generative/dcgan  
https://acdc.creatis.insa-lyon.fr/description/databases.html  
https://github.com/Hadrien-Cornier/E6040-superresolution-project/blob/master/main.ipynb   
https://arxiv.org/ftp/arxiv/papers/1707/1707.05425.pdf    
https://arxiv.org/pdf/2003.01217.pdf    
https://arxiv.org/abs/1608.06993v5    
https://arxiv.org/pdf/1609.04802v5.pdf    
