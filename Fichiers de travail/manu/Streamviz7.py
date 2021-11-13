# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:59:44 2021

@author: utilisateur
"""
import re
import string
import os
import time
import nltk
from nltk import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
#from wordcloud import WordCloud
import streamlit as st
# import altair as alt
from PIL import Image
import tensorflow as tf
from sklearn.utils import shuffle
import base64
#import graphviz as graphviz
import cv2

st.title('RAKUTEN WEB APP FOR MASTER DATASCIENTEST')
st.markdown("""
* WELCOME TO RAKUTEN STREAMLIT APPLICATION
""")

df = pd.read_csv('X_train_update.csv',index_col=0)

designation = df['designation']
description = df['description']

### TEST LE 31 08 (boutons barre de gauche)
with st.sidebar:
            
            st.title("RAKUTEN PROJECT")
            st.sidebar.subheader('DATAVIZ Designation & Description')
            st.subheader("Select a page below:")
            mode = st.radio(
                "Menu",
                [
                    "DataViz",
                    "Modelisation",
                    "Model Game",
                    "A little theory"
                ],
            ) 
            #st.sidebar.header('User Data Selection')
            st.sidebar.subheader('GAME ON DATAS : Target')
        
if mode == "DataViz" :      
            # ETUDE DES NAN'S
            st.header('Observation of NAN values')     
            figW=plt.figure() 
            sns.heatmap(df.isna(),cmap="coolwarm", center = 10.0, cbar=False) #(bins = 100, color='gray')    
            st.pyplot (figW)
            
            #sorted_text_data = st.sidebar.multiselect(designation, description)
            # df_selected_sector = df[df.isin(sorted_text_data)]
            if st.button('Show DATA Designation'):
                st.write(designation)
                
            if st.button('Show DATA Description'):    
                st.write(description)
            
            st.header('Length of Words for RAKUTEN DATASET FEATURES')
            st.write()
           
            # ETUDE DE LA COLONNE DESIGNATION
            
            df['designation'] = df['designation'].apply(lambda _: str(_))
            words_per_review = df.designation.apply(lambda x: len(x.split(" ")))
            #if st.button('Show Plot Designation'):
                #st.header('Longeur des mots de la variable description')    
                #st.area_chart(df.designation.apply(lambda x: len(x.split(" "))))
                
            if st.button('Show Plot Designation'):
                st.header('Length of words for Designation')     
                figx=plt.figure() 
                sns.histplot (words_per_review, kde=True,binwidth = 10, color='black') #(bins = 100, color='gray')
                plt.xlim(0,200)    
                st.pyplot (figx)
                 
            df['description'] = df['description'].apply(lambda _: str(_))
            words_per_review2 = df.description.apply(lambda x: len(x.split(" ")))
            
            #if st.button('Show Plot Description'):
            #    st.header('Longeur des mots de la variable description')    
            #    st.area_chart(df.description.apply(lambda x: len(x.split(" "))))    
            if st.button('Show Plot Description'):
                st.header('Length of words for Description') 
                figy=plt.figure() 
                sns.histplot (words_per_review2, kde=True, binwidth = 10, color='black') #(bins = 100, color='gray')
                plt.xlim(0,200)    
                st.pyplot (figy)
                 
            #if st.button('Show Plots'):
                #st.header('Longeur des mots de la variable description')
                #st.pyplot (figx)
                
            target = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)
            st.header('Study of target RAKUTEN ')
            st.write() 
            df['target'] = target 
            
            if st.button('Show Plot Target_Bar'):
                figz2=plt.figure() 
                percent_target = 100 * df['target'].value_counts()/len(df)
                percent_target.plot.bar(color='gray')
                sns.set_palette('Accent')
                font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 14}            
                plt.rc('font', **font)           
                plt.title("Barplot of different product codes")
                plt.ylabel('% of all targets')
                st.pyplot (figz2)
                          
            if st.button('Show Plot Target_Pie'):
                figa=plt.figure(figsize=(18,12)) 
                df['target'].value_counts().plot.pie()  
                plt.xticks(rotation=45)
                st.pyplot (figa)
                
                             ###################### 
            #df['target'] = target
            ##percent_target = 100 * df['target'].value_counts()/len(df)
            #percent_target = df['target'].value_counts(sort=True, ascending=True)
            #if st.button('Show Plot Target2'):
                #figz=plt.figure() 
                #sns.barplot (x=df['target'], y=percent_target, color='gray') #(bins = 100, color='gray')  
                #plt.xticks(rotation=90)
                #st.pyplot (figz)              
                
                            ###################### 
            #count = df['target'].value_counts(ascending=True)
            #df2=pd.DataFrame(count,columns=['target'])
            #df2= df2.reset_index()
            #df2.index.rename('index_name', inplace=True)
                 
                             ###################### 
            X = pd.read_csv("X_train_update.csv")[:100]
            path = os.getcwd() + '/images/image_train'
            X['image_name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
            X['image_path'] = path + r'/image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
                                             
            st.header('Observation of some IMAGES DATA')
            if st.button('Show Random image of dataset'):
                i = np.random.choice(len(X), size = 1) #Random product to show        
                image = Image.open(path + '/' + X.iloc[i]['image_name'].item())
                st.image(image,'image RAKUTEN DATASET')
                
             #### CODE EN COURS  
            y_train = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)
            X['image_name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
            X['image_path'] = path + r'/image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
            X['y_train'] = y_train
            X.head()
                           
                        
            if st.button('Show Several images of dataset'):
                for cls in list(set(y_train['prdtypecode'].tolist())):
                        pic_path_list = X[y_train['prdtypecode']==cls]['image_name'].tolist()
                        i=0
                
                        for pic in pic_path_list:
                            picture = plt.imread(path + '/' + pic)
                            plt.figure() #figsize = (6,6)
                            plt.axis('off')
                            plt.ylabel(cls)
                            #plt.imshow(picture)
                            st.image(picture)
                            i +=1
           
                
            ##############
           
elif mode == "Modelisation":
    st.title("ANALYSE DU MODELE")
    st.write("MODELE RAKUTEN")
    from rakuten_demo import *
    create_demo()
    cat = CatModel()

    # Affichage de l'architecture
    #########
    cat.model.summary()
    #st.header('Observation of MODELE ARCHITECTURE')
    #if st.button('Show Images'):
        #figb=plt.figure() 
        #figb=tf.keras.utils.plot_model(cat.model, show_shapes = True, show_layer_names = True)
        #st.pyplot (figb)
        
     # Lecture de la dataframe X et affichage des premières lignes
    df = d_get_X_df()
    print("Shape de X =", df.shape)
    if st.button('Show DATA Designation'):
        st.write(df.head())
    
    # Lecture de la Serie des y (de même longueur que les X)
    ys = d_get_y()
    
    # Lecture des données preprocessées pour les modèles texte
    X_text = d_get_X_text()         # designation + description
    X_text_embed_multilingual = d_get_X_text_embed_multilingual()
    X_text_spacy_lemma = d_get_X_text_spacy_lemma()
    if st.button('X_test_spacy_lemma'):
        st.write(X_text_spacy_lemma[0:100])
        
    nb = 200 # Nombre d'échantillons
    idx = np.random.randint(0, nb)
    
    print(f"Classe {ys[idx]}")
    fichier_image = d_get_image(df.iloc[idx])
    image = plt.imread(fichier_image)
    
#### CODE A REPRENDRE ICI
    
    
    

    





                    
elif mode == "Model Game":
    st.title("Model Game")
    st.write("CHOOSE THE SELECTED LABEL OF IMAGE")
    
   
    # CODE EDGAR#####
    
    st.sidebar.write('SELECTED CLASS:')
    # Initialization
    #if 'key' not in st.session_state:
    #    st.session_state['key'] = 'value'
    
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
       '1281':' Jeux société pour enfants, Boites et autres, couleurs flashy',
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

    Livres = {
            #Livres
            '10':' Livres type romain, Couvertures de livres ',
           '2280':' Livres, journaux et revues anciennes',
           '2403':' Livres, BD et revues de collection',
           '2705':' Livres en général',
           '2522':' Cahiers, carnets, marque pages'
        
        }
        
    Jeux = {
            #Jeux
           '40':' Jeux videos, CDs + mais aussi equipements, cables, etc. ',
           '50':' Equipements/ complements consoles, gamers ',
           '2905':' Jeux vidéos pour PC',
           '2462':' Equipement jeux, jeux video, play stations',
           '60':' Consoles '
        }
        
    Jouets = {
            #Jouets & Figurines
           '1280':' Jouets pour enfants, poupées nounours, equipements enfants',
           '1281':' Jeux socitété pour enfants, Boites et autres, couleurs flashy',
           '1300':' Jeux techniques, Voitures/ drones télécomandés, Equipement, petites machines ',
           '1180':' Figurines et boites ',   
           '1140':' Figurines, Personnages et objets, parfois dans des boites ',
            '1160':' Cartes collectionables, Rectangles, beaucoup de couleurs '
          }
        
    Meubles = {
            #Meubles & Equipements
           '1320':' Matériel et meubles bébé poussettes, habits',
           '1560':' Meubles, matelas canapés lampes, chaises',
            '2582':' Matériel, meubles et outils pour le jardin',
           '2583':' Equipements technique pour la maison et exterieur (piscines), produits',
           '2585':' Idem 2583:  Equipements technique pour la maison et exterieur (piscines), produits',
            '1302':' Equipements, Habits, outils, jouets, objets sur fond blanc',
            '2220':' Equipements divers pour animaux'
            }
        
    Deco = {
            #Déco
           '1920':' Oreillers, coussins, draps',
           '2060':' Décorations'
        }
        
    Autre ={
            #Autre
            '1301':' Chaussetes bébés, petites photos ',
           '1940':' Alimentations, conserves boites d gateaux'
        
          }
        
    classes = [Livres,Jeux,Jouets,Meubles,Deco,Autre]
    classes_name = ['LIVRES','JEUX','JOUETS','MEUBLES','DECO','AUTRE']
        
        ##INPUT DATA
        
    path = os.getcwd() + '/images/image_train'
        
    X = pd.read_csv('X_train_update.csv',index_col=0)
    y = pd.read_csv('Y_train_CVw08PX.csv',index_col=0).squeeze().map(str)
    
    X=X[0:2000]
    y=y[0:2000]
        
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
                                
elif mode == "A little theory":
    
        st.title("Theory of CNN")

        st.write(" Sources : Exemple of efficient CNN pretrained model" 
                 "https://keras.io/api/applications/efficientnet/. "
                 "Exemple of API CONCATENATE OF MODELS" 
                 "https://keras.io/api/layers/merging_layers/concatenate/.")
        
        if st.button('SIMPLE CONVOLUTION'):
                        
                original = Image.open('CNN.png')
                st.image(original, use_column_width=True)
                image1 = cv2.imread("CNN.png")

                file_ = open("conv_operation.gif", "rb")
                contents = file_.read()
                data = base64.b64encode(contents).decode("utf-8")
                file_.close()
                
                st.markdown(f'<img src="data:image/gif;base64,{data}" alt="conv_operation.gif">',
                unsafe_allow_html=True,)
                                
                #original = Image.open('conv_operation.gif')
                #               st.image(original, use_column_width=True)
                #                image2 = cv2.imread("conv_operation.gif")
                    
        if st.button('EFFNET B1'):
         
                original = Image.open('EFFNETB1.png')
                st.image(original, use_column_width=True)
                image2 = cv2.imread("EFFNETB1.png")
                
                
        if st.button('MODELE CONCAT RAKUTEN'):
         
                original = Image.open('DETAILCONCAT.png')
                st.image(original, use_column_width=True)
                image3 = cv2.imread("DETAILCONCAT.png")

        
        
        




    
    
    
    
    
    
    
