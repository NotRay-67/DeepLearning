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
These lines import required libraries and modules, including pandas for data manipulation, glob for file searching, matplotlib for plotting, numpy for numerical computations, TensorFlow for deep learning, and Keras for building neural network models. The ImageDataGenerator from Keras is imported to generate batches of augmented data from image files.

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
This part downloads a dataset from Kaggle using the Kaggle API. It creates a Kaggle directory, uploads Kaggle API credentials, downloads the dataset, and extracts it to the "Cat-Dog" directory. It then prints the number of images in the dataset.

### 
```python
CATS = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
cats_images = []
dogs_images = []
for img in glob.glob('/content/Cat-Dog/images/*.jpg'):
  if any(cat in img for cat in CATS):
    cats_images.append(img)
  else:
    dogs_images.append(img)
print('There are {} images of cats'.format(len(cats_images)))
print('There are {} images of dogs'.format(len(dogs_images)))
```
This code segment categorizes images into cats and dogs based on their file names. It prints the number of images for each category.

### Data Splitting

```python
np.random.shuffle(cats_images)
np.random.shuffle(dogs_images)
#split the data into train, validation and test sets
train_d, val_d, test_d = np.split(dogs_images, [int(len(dogs_images)*0.7), int(len(dogs_images)*0.8)])
train_c, val_c, test_c = np.split(cats_images, [int(len(cats_images)*0.7), int(len(cats_images)*0.8)])
train_dog_df = pd.DataFrame({'image':train_d, 'label':'dog'})
val_dog_df = pd.DataFrame({'image':val_d, 'label':'dog'})
test_dog_df = pd.DataFrame({'image':test_d, 'label':'dog'})
train_cat_df = pd.DataFrame({'image':train_c, 'label':'cat'})
val_cat_df = pd.DataFrame({'image':val_c, 'label':'cat'})
test_cat_df = pd.DataFrame({'image':test_c, 'label':'cat'})
train_df = pd.concat([train_dog_df, train_cat_df])
val_df = pd.concat([val_dog_df, val_cat_df])
test_df = pd.concat([test_dog_df, test_cat_df])
print('There are {} images for training'.format(len(train_df)))
print('There are {} images for validation'.format(len(val_df)))
print('There are {} images for testing'.format(len(test_df)))

```
This part shuffles the lists of cat and dog images, then splits them into training, validation, and test sets. Dataframes are created from these splits, with image paths and labels ('dog' or 'cat'). Finally, it prints the number of images in each split.

### Data Preproccesing
```python
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
#rescale the images
trainGenerator = ImageDataGenerator(rescale=1./255.)
valGenerator = ImageDataGenerator(rescale=1./255.)
testGenerator = ImageDataGenerator(rescale=1./255.)
#convert them into a dataset
trainDataset = trainGenerator.flow_from_dataframe(
dataframe=train_df,
class_mode="binary",
x_col="image",
y_col="label",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
target_size=(IMG_HEIGHT,IMG_WIDTH) #set the height and width of the images
)
valDataset = valGenerator.flow_from_dataframe(
dataframe=val_df,
class_mode='binary',
x_col="image",
y_col="label",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
target_size=(IMG_HEIGHT,IMG_WIDTH)
)
testDataset = testGenerator.flow_from_dataframe(
dataframe=test_df,
class_mode='binary',
x_col="image",
y_col="label",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
target_size=(IMG_HEIGHT,IMG_WIDTH)
)

```
This section preprocesses the images. It defines batch size and image dimensions. ImageDataGenerator objects are created for training, validation, and test sets to rescale images. Then, the `flow_from_dataframe()` method is used to convert dataframes into datasets, specifying parameters like batch size, target size, and column names for images and labels.

### 
```python
images, labels = next(iter(testDataset))
print('Batch shape: ', images.shape)
print('Label shape: ', labels.shape)
```
This line extracts a batch of images and their labels from the test dataset using the `next()` and `iter()` functions, then prints their shapes.

###

```python
plt.imshow(images[3])
print('Label: ', labels[3])

plt.imshow(images[5])
print('Label: ', labels[5])
```
These lines display two images from the batch along with their labels.

### Model Building

```python

model = keras.Sequential([
keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
keras.layers.Conv2D(64, (3, 3), activation='relu'),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Conv2D(128, (3, 3), activation='relu'),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Conv2D(256, (3, 3), activation='relu'),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Conv2D(512, (3, 3), activation='relu'),
keras.layers.GlobalAveragePooling2D(),
keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

```
This part builds the neural network model using Keras' Sequential API. It consists of convolutional layers followed by max-pooling layers for feature extraction and downsampling. Finally, it includes a global average pooling layer and a dense layer with a sigmoid activation function for binary classification.

### Model Training

```python
#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs=15
#train the model
history = model.fit(trainDataset, epochs=epochs, validation_data=(valDataset))
```
This segment compiles the model using binary cross-entropy loss and the Adam optimizer. It then trains the model on the training dataset for a specified number of epochs while using the validation dataset for validation during training.

### Model Visualization

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()
```
This section plots the training and validation accuracies over epochs to visualize the model's training performance.

### Model Evaluation
```python
#evaluate the model on the test dataset
loss, acc = model.evaluate(testDataset)

print('Loss:', loss)
print('Accuracy:', acc)
img = plt.imread('/content/dog_image.jpeg')
plt.imshow(img)
```
This part evaluates the trained model on the test dataset and prints the loss and accuracy. the last two line loads and displays an image named "dog_image.jpeg".

## Improvements that can be done 

- Data Augmentation: Augmenting the dataset can improve model generalization. Techniques like rotation, flipping, and zooming can be applied to generate more diverse training examples.
- Model Architecture: Experimenting with different architectures, such as deeper networks or using pre-trained models like ResNet or VGG, may improve performance.
- Hyperparameter Tuning: Tuning hyperparameters like learning rate, batch size, and number of epochs can lead to better model performance.
- Regularization: Applying techniques like dropout or L2 regularization can help prevent overfitting.
- Early Stopping: Implementing early stopping based on validation loss can prevent overfitting and improve generalization.
- Class Imbalance Handling: If there's a significant class imbalance, techniques like class weighting or oversampling can be applied to handle it.
- Learning Rate Scheduling: Implementing learning rate schedules such as decay or adaptive learning rates can help in faster convergence and better performance.

## Conclusion 
the provided code demonstrates a binary classification deep learning model trained to distinguish between images of cats and dogs. It fetches a dataset from Kaggle, preprocesses the images, builds and trains a convolutional neural network (CNN) model, evaluates its performance, and visualizes training metrics.

However, there are several areas for improvement to enhance the model's performance and generalization. These include implementing data augmentation techniques, experimenting with different model architectures, tuning hyperparameters, applying regularization methods, handling class imbalances, and optimizing learning rate schedules.


