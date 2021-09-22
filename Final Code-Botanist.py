#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[2]:


df = pd.read_csv("./Botanist_Training_Set.csv")


# In[3]:


def append_ext(fn):
    return str(fn) +".jpg"
df["FileName"] = df["FileName"].apply(append_ext)
df.head()


# In[4]:


df['FileName'] = df['FileName'].astype('str')
df['Label'] = df['Label'].astype('str')


# In[5]:


datagen = ImageDataGenerator(
    width_shift_range= 0.2, height_shift_range= 0.2,
    rotation_range= 90, rescale = 1/255,validation_split=0.25,
    horizontal_flip= True, vertical_flip=True)

ValidData = ImageDataGenerator(
    horizontal_flip=True)

columns = ['FileName', 'Label']    

train_generator = datagen.flow_from_dataframe(
            dataframe=df, directory="./TrainFiles/", 
            x_col="FileName", y_col= 'Label', 
            class_mode="categorical", target_size=(256,256), 
            batch_size= 64, validate_filenames=False, subset = "training")

valid_generator = datagen.flow_from_dataframe(
            dataframe=df, directory="./TrainFiles/", 
            x_col="FileName", y_col= 'Label', 
            class_mode="categorical", target_size=(256,256), 
            batch_size=64, validate_filenames=False, subset = "validation")


# In[6]:


num_classes = 38

model = Sequential([
  keras.layers.Conv2D(32, (5,5),activation='relu', input_shape=(256,256, 3)),
  keras.layers.MaxPooling2D(pool_size=(4,4)), 
    
  keras.layers.Conv2D(64, (3,3), strides = (1,1),activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2,2)),
    
  keras.layers.Conv2D(128, (3,3), strides = (1,1),activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2,2)),

keras.layers.Conv2D(256, (3,3), strides = (2,2),activation='relu'),   
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.5),  
      keras.layers.Dense(64, activation='relu'),
  keras.layers.Dropout(0.3),  
  keras.layers.Dense(num_classes, activation='softmax')
])


# In[7]:


model.compile(optimizer= "Adam", 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


model.summary()


# In[10]:


model.fit(train_generator, validation_data = valid_generator, epochs = 10, verbose=1)


# In[11]:


model.evaluate(valid_generator)


# In[12]:


model.save("./test10.h5")


# In[ ]:




