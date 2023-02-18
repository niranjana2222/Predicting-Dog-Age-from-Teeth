from pathlib import Path
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib as plt
import os
import tensorflow as tf
import seaborn as sns

gpus = tf.config.experimental.list_physical_devices('GPU')



if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
tf.config.experimental.set_visible_devices([], 'GPU')

#use command :  conda activate tensorflow_env

img_path = Path("dog")

#create a dataframe of the ages
df = pd.read_csv("age.csv")
df = df.drop(columns=["N"])

#group into 0-2, 2-5, 5-10, 10-infinity
df["age"] = pd.cut(df["Age"],bins=[-1, 2, 5, 10, np.inf],labels=[1, 2, 3, 4])
#df["age"] = df["Age"]
#age = df["age"]
#print(df["age"].value_counts())
df = df.drop(columns=["Age"])
print(df)
age = df["age"]

times = 2
"""
for index, row in df.iterrows():
    
    line = pd.DataFrame({"Breed": row["Breed"], "age": row["age"]}, index=[index+1])
    #for i in range(0, times):
    df = pd.concat([df[:index], line, line, df[index:]]).reset_index(drop=True)
"""

df2 = pd.DataFrame(columns = df.columns)
for index, row in df.iterrows():
    print(df.iloc[index])
    for i in range(0,times+1):
        df2 = df2.append(df.iloc[index])
print(df2)
df = df2

#create the dataset
imgs = sorted(list(img_path.glob('*'))) #creates a list of the paths of the images

filepaths = pd.Series(imgs, name='Filepath').astype(str)
labels = pd.Series(age, name='Label')

import random
class Data_augmentation:
    def __init__(self, path, image_name):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.path = path
        self.name = image_name
        print(path+image_name)
        self.image = cv2.imread(path+image_name)

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image
    
    
    def image_augment(self, save_path):
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        img = self.image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img)
        #img_gaussian = self.add_GaussianNoise(img)
        
        name_int = self.name[:len(self.name)-4]
        cv2.imwrite(save_path+'%s' %str(name_int)+'_vflip.jpg', img_flip)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_rot.jpg', img_rot)
        #cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
    
    
    def main(file_dir,output_path):
        for root, _, files in os.walk(file_dir):
            print(root)
        for file in files:
            raw_image = Data_augmentation(root,file)
            if file == ".DS_Store":
                continue
            #print(file)
            raw_image.image_augment(output_path)
    
images = []
#images = pd.DataFrame(filepaths)
Data_augmentation.main('dog/', 'dog/')
    
for filename in os.listdir('dog'):
    if filename == ".DS_Store":
        continue
    image = cv2.imread(os.path.join('dog', filename))
    #print(os.path.join('dog', filename))
    image = cv2.resize(image, (224, 224))
    images.append(image)

images = np.array(images)
#images = pd.concat([filepaths, df["Breed"], labels], axis=1)
#print(images.Label.value_counts())

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
    
    # return our model
    return model
    
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model
    
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

#print(df.shape)
#print(images.shape)
split = train_test_split(df, images, test_size=0.25, random_state=42)
(train_val, test_val, train_images, test_images) = split
train_y = train_val["age"]
test_y = test_val["age"]

train_val = train_val.drop(columns=["age"])
test_val = test_val.drop(columns=["age"])

print("bred", train_val)
print("ans", train_y)
# create the MLP and CNN models
mlp = create_mlp(1, regress=False)
cnn = create_cnn(224, 224, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(5, activation="relu")(combinedInput)
x = Dense(4, activation="softmax")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)
# train the model
print("[INFO] training model...")
history = model.fit(
    x=[train_val, train_images], y=train_y,
    validation_data=([test_val, test_images], test_y),
    epochs=5, batch_size=1)


results = model.evaluate(x=[test_val, test_images], y= test_y, verbose=1)
#print("Real", test_y)
#print("Predict", np.argmax(results), axis=1)
print(f"Test Accuracy: {np.round(results)}%")

#confusion matrix
import matplotlib.pyplot as plt

predictions = np.argmax(model.predict([test_val, test_images]), axis=1)

from sklearn.metrics import accuracy_score

#print(accuracy_score(test_y, np.argmax(predictions, axis=1)))

from sklearn.metrics import confusion_matrix, classification_report

print("test ", test_y)
print("pred", predictions)

matrix = confusion_matrix(test_y, predictions)
fig = plt.figure(figsize=(5, 5))
sns.heatmap(matrix, annot=True, cmap='viridis')
plt.xticks(ticks=np.arange(5) + 0.5, labels=test_y, rotation=90)
plt.yticks(ticks=np.arange(5) + 0.5, labels=test_y, rotation=0)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fig.savefig("Confusion Matrix",dpi=700)

report= classification_report(test_y, predictions,  zero_division=0)
print("Classification Report:\n", report)

#loss graph
data_his = pd.DataFrame(history.history)

plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 4))
plt.plot(data_his['loss'], label = 'train')
#plt.plot(data_his['val_loss'], label = 'val')
plt.legend()
plt.title('Loss Function')
plt.show()
fig.savefig("Loss Function",dpi=700)

"""
#accuracy graph
fig = plt.figure(figsize=(18, 4))
plt.plot(data_his['accuracy'], label = 'train')
#plt.plot(data_his['val_accuracy'], label = 'val')
plt.legend()
plt.title('Accuracy Function')
plt.show()
fig.savefig("Accuracy Function",dpi=700)
"""

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
