#!/usr/bin/env python
# coding: utf-8

# # DEEP LEARNING - Itération 1

# # Partie 1 : Images

# ### Objectif à dépasser : weighted F1-score = 0.5534 (Resnet)

# Import libraries:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\image_ds\\images')


# Récupérer les données:

# In[2]:


X = pd.read_csv('X_train_update.csv',index_col=0)
y = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)


# Liste des images:

# In[3]:


os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\image_ds')


# In[4]:


import os #Miscellaneous operating system interfaces
#https://docs.python.org/3/library/os.html

#get current working directory
current_path = os.getcwd() 

#Training images path
images_path = current_path + r'/images/image_train/'

#List with the name of all training images
images_list = os.listdir(images_path)


# Géneration nom des images:

# In[5]:


#Create a column with the name of the picture
X['image name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
X['image name']


# ### Répartition des images dans les échantillons train, validation et test

# Répartion :

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 123)


# Qualité de la répartitioon

# In[7]:


trainy = pd.DataFrame(y_train.value_counts(), columns = ['Nombre_images_train'])
validy = pd.DataFrame(y_val.value_counts(), columns = ['Nombre_images_validation'])
testy = pd.DataFrame(y_test.value_counts(), columns = ['Nombre_images_test'])
train_valid = trainy.merge(validy, right_index = True, left_index = True)
train_valid_test = train_valid.merge(testy, right_index = True, left_index = True)
train_valid_test['y'] = train_valid_test.index
yval = []
for i in train_valid_test['y']:
    yval.append(i[0])
train_valid_test['y'] = yval
train_valid_test


# In[8]:


train_valid_test.index.tolist()[0]


# In[9]:


np.max(train_valid_test.Nombre_images_train.tolist())


# In[10]:


train_valid_test.plot.bar("y", "Nombre_images_train") 
train_valid_test.plot.bar("y", "Nombre_images_validation")
train_valid_test.plot.bar("y", "Nombre_images_test")


# ### Processus de génération des données

# #### APPLICATION CNN

# "A rough rule of thumb is that you need at least 1,000 images from each class you are trying to classify… and more is always better", https://medium.com/@bcwalraven/boost-your-cnn-with-the-keras-imagedatagenerator-99b1ef262f47

# In[11]:


# On a pas dans la base de données complète (tous échantillons confondus) pour chaque classe 1000 images
# ==> Augmentation des données
y.value_counts().min()


# In[12]:


from keras.utils import np_utils 

y_train_cnn = np_utils.to_categorical(y_train)
y_val_cnn = np_utils.to_categorical(y_val)
y_test_cnn = np_utils.to_categorical(y_test)


# In[13]:


X_train['class'] = y_train
X_val['class'] = y_val
X_test['class'] = y_test


# The generator will run through your image data and apply random transformations to each individual image as it is passed to the model so that it never sees the exact same image twice during training. These transformations are parameters on the generator that can be set when instantiated and can include rotations, shears, flips, and zooms. The benefit here is that the model will become more robust as it trains on images that are slightly distorted, and it helps to prevent the model from learning noise in your data such as where features are located in the image. 

# Image Data Generator :

# In[14]:


from keras.preprocessing.image import ImageDataGenerator
train_data_generator = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split = .2)
val_data_generator = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split = .2) # Données sur lesquelles améliorer le modèle
test_data_generator = ImageDataGenerator(rescale=1./255) # Données sans entraînement, juste pour test


# Directory's changement for the data access :

# In[15]:


path = os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\image_ds\\images\\image_train')


# Iteration process : 

# In[16]:


batch_size = 32

X_train["class"] = X_train["class"].astype(str)
X_val["class"] = X_val["class"].astype(str)
X_test["class"] = X_test["class"].astype("str")

train_generator = train_data_generator.flow_from_dataframe(dataframe=X_train,
                                                          directory=path,
                                                           x_col = "image name",
                                                           y_col = "class",
                                                           class_mode ="sparse",
                                                          target_size = (128, 128), 
                                                          batch_size = batch_size)
val_generator = train_data_generator.flow_from_dataframe(dataframe=X_val,
                                                          directory=path,
                                                           x_col = "image name",
                                                           y_col = "class",
                                                           class_mode ="sparse",
                                                          target_size = (128, 128), 
                                                          batch_size = batch_size)

# Remarque test à ne pas toucher jusqu'à l'évaluation finale du modèle, ajout shuffle = False ?

test_generator = train_data_generator.flow_from_dataframe(dataframe=X_test,
                                                          directory=path,
                                                           x_col = "image name",
                                                           y_col = "class",
                                                           class_mode ="sparse",
                                                          target_size = (128, 128), 
                                                          batch_size = batch_size)


# Résumé données image : 

# In[17]:


batchX, batchy = train_generator.next()
print('Batch shape (taille batch, shape image) = %s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


# # CNN

# Package modélisation CNN :

# In[18]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils 


# (#) Remarque : on  peut inclure dans la métrique de la compilation le F1_WEIGHTED
# 
# 
# import keras.backend as K
# 
# def f1_weighted(true, pred): #shapes (batch, 4)
# 
#     #for metrics include these two lines, for loss, don't include them
#     #these are meant to round 'pred' to exactly zeros and ones
#     predLabels = K.argmax(pred, axis=-1)
#     pred = K.one_hot(predLabels, 4) 
# 
# 
#     ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
#     pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
#     true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
#         #all with shape (4,)
#     
#     precision = true_positives / pred_positives 
#     recall = true_positives / ground_positives
#         #both = 1 if ground_positives == 0 or pred_positives == 0
#         #shape (4,)
# 
#     f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
#         #still with shape (4,)
# 
#     weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
#     weighted_f1 = K.sum(weighted_f1)
# 
#     
#     return weighted_f1

# Création du modèle :

# In[19]:


model = Sequential()

first_layer = Conv2D(filters = 32,
                     kernel_size = (5, 5),
                     padding = 'valid',
                     input_shape = (128, 128, 3),
                     activation = 'relu')

second_layer = MaxPooling2D(pool_size = (2, 2))

model.add(first_layer)
model.add(second_layer)

third_layer = Dropout(rate = 0.2)
fourth_layer = Flatten()
fifth_layer = Dense(units = 128, activation = "relu")
output_layer = Dense(units = 27, activation = "softmax")
model.add(third_layer)
model.add(fourth_layer)
model.add(fifth_layer)
model.add(output_layer)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["acc"])


# Résumé du modèle :

# In[20]:


model.summary()


# Entrainement  (Epochs = 2) :

# In[21]:


history = model.fit(train_generator, epochs = 2, steps_per_epoch = len(X_train)//batch_size, validation_steps = len(X_train)//32, validation_data = val_generator)


# Prédiction : 

# In[22]:


y_pred_2 = model.predict(test_generator, verbose = 1)
y_pred_2 = y_pred_2.argmax(axis = 1)


# In[23]:


y_pred_2 = y_pred_2.argmax(axis = 1)


# Evaluation (F1-score) :

# In[24]:


from sklearn.metrics import f1_score
f1_score(y_pred_2, y_test, average = "weighted")


# Correspondance entre les labels :

# In[25]:


labels = (train_generator.class_indices)
print(labels)


# Entrainement (Epochs = 10) :

# In[ ]:


history = model.fit_generator(generator = train_generator, epochs = 10, steps_per_epoch = len(X_train)//batch_size, validation_steps = len(X_train)//32, validation_data = val_generator)


# In[ ]:


y_pred_10 = model.predict(test_generator,verbose=1,steps=len(X_test)//batch_size)
y_pred_10 = np.argmax(y_pred_10, axis=1)


# Evaluation (F1-score) :

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_pred, y_test, average = "weighted")

