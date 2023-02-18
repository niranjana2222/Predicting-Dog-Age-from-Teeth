from pathlib import Path
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib as plt
import os
import tensorflow as tf
import seaborn as sns
#use command :  conda activate tensorflow_env

img_path = Path("dog")

#create a dataframe of the ages
df = pd.read_csv("copy.csv")

#group into 0-2, 2-5, 5-10, 10-infinity
df["age"] = pd.cut(df["Age"],bins=[-1, 2, 5, 10, np.inf],labels=["0-2", "2-5", "5-10", "10-20"])
age = df["age"]
#print(df["age"].value_counts())

#create the dataset
imgs = sorted(list(img_path.glob('*'))) #creates a list of the paths of the images

filepaths = pd.Series(imgs, name='Filepath').astype(str)
labels = pd.Series(age, name='Label')

images = pd.concat([filepaths, labels], axis=1)
print(images.Label.value_counts())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model


#train test split
import sklearn
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

#process the image data
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2,  rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, brightness_range=[0.2,1.2])
    

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    
#create the training and validation data
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=2,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=2,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=2,
    shuffle=False
)

#load the pretrained model
pretrained_model = tf.keras.applications.resnet50.ResNet50(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')

pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
#print(model.summary())

#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
history = model.fit(train_images,validation_data=val_images,epochs=5,  verbose=1)

results = model.evaluate(test_images, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100,2)}%")

#training history
data_his = pd.DataFrame(history.history)
print(data_his)

#confusion matrix
import matplotlib.pyplot as plt

predictions = np.argmax(model.predict(test_images), axis=1)

from sklearn.metrics import confusion_matrix, classification_report

"""
matrix = confusion_matrix(test_images.labels, predictions)
fig = plt.figure(figsize=(30, 30))
sns.heatmap(matrix, annot=True, cmap='viridis')
plt.xticks(ticks=np.arange(20) + 0.5, labels=test_images.class_indices, rotation=90)
plt.yticks(ticks=np.arange(20) + 0.5, labels=test_images.class_indices, rotation=0)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fig.savefig("Confusion Matrix",dpi=700)
"""
report= classification_report(test_images.labels, predictions, target_names=test_images.class_indices, zero_division=0)
print("Classification Report:\n", report)

#loss graph
plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 4))
plt.plot(data_his['loss'], label = 'train')
plt.plot(data_his['val_loss'], label = 'val')
plt.legend()
plt.title('Loss Function')
plt.show()
fig.savefig("Loss Function",dpi=700)

#accuracy graph
fig = plt.figure(figsize=(18, 4))
plt.plot(data_his['accuracy'], label = 'train')
plt.plot(data_his['val_accuracy'], label = 'val')
plt.legend()
plt.title('Accuracy Function')
plt.show()
fig.savefig("Accuracy Function",dpi=700)

"""
# Convert sklearn model to CoreML
import coremltools as cml

model = cml.converters.sklearn.("IMAGE","AGE")

# Assign model metadata
model.author = "Niranjana Sankar"
model.license = "N/A"
model.short_description = "Dog age estimation"

# Assign feature descriptions
model.input_description["IMAGE"] = "Image of dog teeth"

# Assign the output description
model.output_description["AGE"] = "Age of dog in 4 categories"

# Save model
model.save('age_estimation.mlmodel')

"""
