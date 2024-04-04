# Title: Binary Classification Deep Learning Model for Cat and Dog Images

## Abstract:
This project implements a deep learning model for binary classification of cat and dog images using TensorFlow and Keras. The dataset used for training and evaluation is the Oxford-IIIT Pet Dataset, consisting of images of various cat and dog breeds. The model architecture comprises convolutional neural network (CNN) layers followed by dense layers for classification. The model is trained, validated, and evaluated to assess its performance.

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
