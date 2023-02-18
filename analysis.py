from pathlib import Path
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib as plt
import os
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
#use command :  conda activate tensorflow_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

mozaic_path = Path("data/mozaics")
masks_path = Path("data/masks")
images_path = Path("data/imgs")

#create a dataframe of the ages
age_path = Path("data/Data.csv")
df = pd.read_csv(age_path)

#group into 0-2, 2-5, 5-10, 10-infinity
df["age"] = pd.cut(df["Age"],bins=[-1, 2, 5, 10, np.inf],labels=["0-2", "2-5", "5-10", "10-20"])
age = df["age"]

#analyze groups
print("Groups", df["age"].value_counts())

#plot a boxplot to show outliers
fig = plt.figure(figsize =(5,5))
df["age"].plot.box
plt.show()
#print(df["age"].info())

print(df["age"].describe())

#analyze original
print("Original", df["Age"].value_counts().sort_index())
df["Age"].plot(kind='box')

#print(df["Age"].info())

print(df["Age"].describe())

