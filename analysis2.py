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

#create a dataframe of the ages
age_path = Path("copy.csv")
df = pd.read_csv(age_path)

#analyze groups
print("Age\n", df["Age"].value_counts())
print("Breed\n", df["Breed"].value_counts())

print(df.describe())

plt.figure()
df["Age"].plot(kind="bar")
plt.show()


