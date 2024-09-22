import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from patchify import patchify
from PIL import Image
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tqdm import tqdm
import random
import pickle
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam

sm.set_framework('tf.keras')

# Function to load image data
def data_loader(folder_dir):
    image_dataset = []
    for images in os.listdir(folder_dir):
        image = cv2.imread(folder_dir + '/' + images, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)
        image = np.array(image)
        image_dataset.append(image)
    return image_dataset

# Function to convert RGB to 2D label
def rgb_to_2D_label(label, class_dict):
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    for i, class_name in enumerate(class_dict['name']):
        label_seg[np.all(label == class_dict.iloc[i, 1:].to_numpy(), axis=-1)] = i
    return label_seg[:, :, 0]

# Get dataset paths from user input
original_images_path = input("Enter the path for the original images: ")
mask_labels_path = input("Enter the path for the mask labels CSV: ")

# Load the datasets
image_dataset = data_loader(original_images_path)  # Real images

image_dataset = np.array(image_dataset)

# Sanity check
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.show()

# Load mask labels
mask_labels = pd.read_csv(mask_labels_path)
print(mask_labels)

# Convert RGB values to label values using the mask labels
labels = []
for i in range(len(image_dataset)):
    label = rgb_to_2D_label(image_dataset[i], mask_labels)
    labels.append(label)
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

# Another sanity check
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

# Prepare data for training
n_classes = len(np.unique(labels))  # Number of classes
labels_cat = to_categorical(labels, num_classes=n_classes)  # One-hot encoding the labels
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

# Preprocess input
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

# Define model
model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
metrics = ['accuracy']
model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

model_resnet_backbone.summary()

# Save the model
model_resnet_backbone.save('resnet_backbone.hdf5')

# Load the model
model = load_model('resnet_backbone.hdf5')

# Make predictions
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

# Randomly select an image from the test set and make prediction
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

# Plot the real image, test labeled image, and predicted labeled image
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
