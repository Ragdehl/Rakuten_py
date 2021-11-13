# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 23:45:35 2021

@author: Edgar
"""

import streamlit as st
import numpy as np
import pandas as pd
import os #Miscellaneous operating system interfaces
import tensorflow as tf

from PIL import Image

all_classes = {
            #Livres
            '10':' Livres type romain, Couvertures de livres ',
           '2280':' Livres, journaux et revues anciennes',
           '2403':' Livres, BD et revues de collection',
           '2705':' Livres en général',
           '2522':' Cahiers, carnets, marque pages',

            #Jeux
           '40':' Jeux videos, CDs + mais aussi equipements, cables, etc. ',
           '50':' Equipements/complements consoles, gamers ',
           '2905':' Jeux vidéos pour PC',
           '2462':' Equipement jeux, jeux video, play stations',
           '60':' Consoles ',

            #Jouets & Figurines
           '1280':' Jouets pour enfants, poupées nounours, equipements enfants',
           '1281':' Jeux socitété pour enfants, Boites et autres, couleurs flashy',
           '1300':' Jeux techniques, Voitures/drones télécomandés, Equipement, petites machines ',
           '1180':' Figurines et boites ',   
           '1140':' Figurines, Personnages et objets, parfois dans des boites ',
            '1160':' Cartes collectionables, Rectangles, beaucoup de couleurs ',
           
            #Meubles
           '1320':' Matériel et meubles bébé poussettes, habits',
           '1560':' Meubles, matelas canapés lampes, chaises',
    
            #Equipements
            '2582':' Matériel, meubles et outils pour le jardin',
           '2583':' Equipements technique pour la maison et exterieur (piscines), produits',
           '2585':' Idem 2583:  Equipements technique pour la maison et exterieur (piscines), produits',
            '1302':' Equipements, Habits, outils, jouets, objets sur fond blanc',
            '2220':' Equipements divers pour animaux',
    
            #Déco
           '1920':' Oreillers, coussins, draps',
           '2060':' Décorations',
    
            #Autre
            '1301':' Chaussetes bébés, petites photos ',
           '1940':' Alimentations, conserves boites d gateaux',

          }

livres = {
    #Livres
    '10':' Livres type romain, Couvertures de livres ',
   '2280':' Livres, journaux et revues anciennes',
   '2403':' Livres, BD et revues de collection',
   '2705':' Livres en général',
   '2522':' Cahiers, carnets, marque pages'

}

jeux = {
    #Jeux
   '40':' Jeux videos, CDs + mais aussi equipements, cables, etc. ',
   '50':' Equipements/ complements consoles, gamers ',
   '2905':' Jeux vidéos pour PC',
   '2462':' Equipement jeux, jeux video, play stations',
   '60':' Consoles '
}

jouets = {
    #Jouets & Figurines
   '1280':' Jouets pour enfants, poupées nounours, equipements enfants',
   '1281':' Jeux socitété pour enfants, Boites et autres, couleurs flashy',
   '1300':' Jeux techniques, Voitures/ drones télécomandés, Equipement, petites machines ',
   '1180':' Figurines et boites ',   
   '1140':' Figurines, Personnages et objets, parfois dans des boites ',
    '1160':' Cartes collectionables, Rectangles, beaucoup de couleurs '
  }

meubles = {
    #Meubles & Equipements
   '1320':' Matériel et meubles bébé poussettes, habits',
   '1560':' Meubles, matelas canapés lampes, chaises',
    '2582':' Matériel, meubles et outils pour le jardin',
   '2583':' Equipements technique pour la maison et exterieur (piscines), produits',
   '2585':' Idem 2583:  Equipements technique pour la maison et exterieur (piscines), produits',
    '1302':' Equipements, Habits, outils, jouets, objets sur fond blanc',
    '2220':' Equipements divers pour animaux'
    }

deco = {
    #Déco
   '1920':' Oreillers, coussins, draps',
   '2060':' Décorations'
}

autre ={
    #Autre
    '1301':' Chaussetes bébés, petites photos ',
   '1940':' Alimentations, conserves boites d gateaux'

  }

classes = [livres,jeux,jouets,meubles,deco,autre]
classes_name = ['livres','jeux','jouets','meubles','deco','autre']

##INPUT DATA

path = os.getcwd() + '/images/image_train'

X = pd.read_csv('X_train_update.csv',index_col=0)
y = pd.read_csv('Y_train_CVw08PX.csv',index_col=0).squeeze().map(str)

#Create a column with the name of the picture
X['image_name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
X['image_path'] = path + r'/image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'

##INPUT DATA

##PAGE PRINCIPALE

st.title('CLASSIFICATION PRODUITS RAKUTEN')

st.write("Essayez de classifier le produit dans la bonne classe")

###Show product
i = np.random.choice(len(X), size = 1) #Random product to show

image = Image.open(path + '/' + X.iloc[i]['image_name'].item())
st.image(image,'text')

st.write('Designation:',X['designation'][i].item())
st.write('Description:',X['description'][i].item())
###Show product

#Create buttons
for classe,name in zip(classes,classes_name):
    st.write(name)
    for label in classe:
        class_man = st.button(classe[label], key=label)

##PAGE PRINCIPALE

##SIDE BAR

st.sidebar.write('Selected class:')

for classe in classes:
    for label in classe:
        if st.session_state[label]: 
            class_man = classe[label]
            st.sidebar.write(class_man)

# Initialization
if 'true_class' not in st.session_state:
    st.session_state['true_class'] = 'value'
    
if class_man == st.session_state['true_class']:
    st.sidebar.write('CORRECT!')
else:
    st.sidebar.write('ERREUR!')
    
st.sidebar.write('Vraie classe:')
#st.sidebar.write(all_classes[y[i].item()])
st.sidebar.write(st.session_state['true_class'])

st.session_state['true_class'] = all_classes[y[i].item()]

##SIDE BAR

@st.cache

def load_tf_model():
    path = r'C:\Users\Edgar\Documents\Rakuten\models_output\Concat\v1\RNN_V5_RNN_V6_EffNetb1_v2.hdf5'
    saved_model = tf.keras.models.load_model(path)
    #saved_model.load_weights(model_path + filename + '.hdf5')
    #saved_model.summary()
    return saved_model

model = load_tf_model()

from keras.applications.efficientnet import preprocess_input

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(#rescale = 1./255,
                                                              preprocessing_function = preprocess_input
                                                              )

path = os.getcwd() + '\\images\\image_train'
'''image_test_set = test_datagen.flow_from_dataframe(dataframe=X.iloc[i],
                                              directory=path,
                                              x_col = "image_name",
                                              y_col = "prdtypecode",
                                            class_mode ="sparse",
                                              target_size = (224, 224),
                                              batch_size = 1,
                                           shuffle=False)
'''
#y_pred_prob = model.predict(X_test)