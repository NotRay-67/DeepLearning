# Title: Binary Classification Deep Learning Model for Cat and Dog Images

## Abstract:
This project implements a deep learning model for binary classification of cat and dog images using TensorFlow and Keras. The dataset used for training and evaluation is the Oxford-IIIT Pet Dataset, consisting of images of various cat and dog breeds. The model architecture comprises convolutional neural network (CNN) layers followed by dense layers for classification. The model is trained, validated, and evaluated to assess its performance.

##  Dataset

This is the `The Oxford-IIIT Pet Dataset`. 
The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.

Source : [The Oxford-IIIT Pet Dataset](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)

## Explaining the Code Blocks:

### Importing Necessary Libraries:
```python
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
```

### Getting the Data:

```python
from google.colab import files
files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d tanlikesmath/the-oxfordiiit-pet-dataset
!unzip /content/the-oxfordiiit-pet-dataset.zip -d Cat-Dog
print('There are {} images in the dataset'.format(len(glob.glob('/content/Cat-Dog/images/*.jpg'))))
```

