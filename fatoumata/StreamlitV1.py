# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:19:14 2021

@author: barry
"""

 # Bibliotheques utilisés:
import pandas as pd
import numpy as np

import streamlit as st

import os #Miscellaneous operating system interfaces
import cv2 #import OpenCV

from sklearn import metrics
import matplotlib.pyplot as plt # Pour l'affichage d'images
from matplotlib import cm # Pour importer de nouvelles cartes de couleur
import itertools # Pour créer des iterateurs


import streamlit as st
import pydot
import graphviz

from PIL import Image

from sklearn.metrics import classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)

###################
#PAGE CONFIGURATION
###################

st.set_page_config(page_title="Projet Rakuten", 
                   page_icon=":robot_face:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

#########
#SIDEBAR
########

new_title = '<p style="font-family:sans-serif; color:RED; font-size: 38px;">Projet RAKUTEN</p>'
st.sidebar.title(new_title,"Projet Rakuten")
st.sidebar.write('')

st.sidebar.markdown("* Promotion Décembre 2020")
st.sidebar.markdown("")
st.sidebar.markdown("## Participants ")
st.sidebar.write("Fatoumata Barry")
st.sidebar.markdown("Emmanuel Bonnet")
st.sidebar.markdown("Edgar Hidalgo")
st.sidebar.markdown("Eric Marchand")
st.sidebar.markdown('')
st.sidebar.markdown('')

st.sidebar.markdown("### ** Sommaire **")

navigation = st.sidebar.radio('',["Introduction", "DataViz", "DataViz - approfondissement", "Méthode", "Démonstration",  "Bilan","Jeu" ])




#CONTACT
########
expander = st.sidebar.expander('SUPPORT')
expander.write("Lien GitHub : https://github.com/Ragdehl/Rakuten_py/tree/main/Livrables/It%C3%A9ration_4 ")



classes = {
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

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nltk
from nltk import word_tokenize
import re
import string

nltk.download('punkt')
from nltk.corpus import stopwords
import unicodedata
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt 
from nltk.stem.snowball import FrenchStemmer
nltk.download('stopwords')
import string



if navigation == "Introduction":
    st.title("Présentation du projet")
    st.title('')
    st.title('')

            
            
    st.write(""" 
 Les techniques de classification « image plus texte » sont fortement sollicitées par les entreprises de e-commerce notamment pour:

 -	La classification de produit à grande échelle pour mieux gérer l’offre d’un site
 -	La recommandation de produit (génère en moyenne 35% de revenu supplémentaire)
 -	L’amélioration de l’expérience client

 Ce projet s’inscrit ainsi dans le challenge Rakuten France Multimodal Product Data Classification. Ce dernier requiert de prédire le code type des produits à partir d’une description texte et d’une image.

 Un modèle de référence est indiqué par le site du challenge: l’objectif du projet est de faire mieux.

 Cette référence est en réalité composée de deux modèles distincts : un pour les images, un pour le texte :

 1.	Pour les données images, une version du modèle Residual Networks (ResNet), le ResNet50 pré-entraîné avec un jeu de données Imagenet; 27 couches différentes du haut sont dégelées, dont 8 couches de convolution pour l'entraînement.
 2.	Pour les données textes, un classificateur RNN simplifié est utilisé. Seuls les champs de désignation sont utilisés dans ce modèle de référence.

    
    
 Les données appartiennent à 27 classes distinctes. Pour évaluer la qualité de la classification, il est demandé d’utiliser la métrique weighted-F1-score. Il s’agit de la moyenne des F1-scores de toutes les classes pondérées par le nombre de représentants dans ces classes. Le F1-score de chaque classe est la moyenne harmonique de la précision et du rappel pour cette classe.

 Le modèle de référence obtient les résultats suivants :
 -	0.5534 pour le modèle image (ResNet)
 -	0.8113 pour le modèle texte (RNN)

    """)        


if navigation == "DataViz" :      
            # ETUDE DES NAN'S
            
            os.chdir('C:/Users/barry/OneDrive - CSTBGroup/ds_images')
            st.title('Exploration des données ')

            df = pd.read_csv('X_train_update.csv',index_col=0)

            designation = df['designation']
            description = df['description']
            st.header('Observation des données texte manquantes')     
            figW=plt.figure() 
            sns.heatmap(df.isna(),cmap="coolwarm", center = 10.0, cbar=False) #(bins = 100, color='gray')    
            st.pyplot (figW)
            
            #sorted_text_data = st.sidebar.multiselect(designation, description)
            # df_selected_sector = df[df.isin(sorted_text_data)]
            if st.button('Show DATA Designation'):
                st.write(designation)
                
            if st.button('Show DATA Description'):    
                st.write(description)
            
            st.header('Longueur des mots des variables TEXTE RAKUTEN ')
            st.write()
           
            # ETUDE DE LA COLONNE DESIGNATION
            
            df['designation'] = df['designation'].apply(lambda _: str(_))
            words_per_review = df.designation.apply(lambda x: len(x.split(" ")))
            #if st.button('Show Plot Designation'):
                #st.header('Longeur des mots de la variable description')    
                #st.area_chart(df.designation.apply(lambda x: len(x.split(" "))))
                
            if st.button('Show Plot Designation'):
                st.header('Longueur des mots de la variable designation - autre graph')     
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
                figy=plt.figure() 
                sns.histplot (words_per_review2, kde=True, binwidth = 10, color='black') #(bins = 100, color='gray')
                plt.xlim(0,200)    
                st.pyplot (figy)
                 
            #if st.button('Show Plots'):
                #st.header('Longeur des mots de la variable description')
                #st.pyplot (figx)
                
            target = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)
            st.header('Etude du target RAKUTEN ')
            st.write() 
            df['target'] = target 
            
            if st.button('Show Plot Target'):
                figa=plt.figure(figsize=(18,12)) 
                df['target'].value_counts().plot.pie()  
                plt.xticks(rotation=45)
                st.pyplot (figa)
                
            df['target'] = target
            ##percent_target = 100 * df['target'].value_counts()/len(df)
            percent_target = df['target'].value_counts(sort=True, ascending=True)
            if st.button('Show Plot Target2'):
                figz=plt.figure() 
                sns.barplot (x=df['target'], y=percent_target, color='gray') #(bins = 100, color='gray')  
                plt.xticks(rotation=90)
                st.pyplot (figz) 
            
            st.header('OBSERVATION OF IMAGES DATA :  SAMPLE ')
            if st.button('Show Images'):
        
                original = Image.open('RAKUTEN_DATAVIZ2.png')
                st.image(original, use_column_width=True)
                image2 = cv2.imread("RAKUTEN_DATAVIZ2.png")


if navigation == "DataViz - approfondissement":
    os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\ds_images')
    st.title('Exploration des données - Approfondissement')
    
    st.title('')
    st.title('')

    X = pd.read_csv('X_train_update.csv',index_col=0)
    y = pd.read_csv('Y_train_CVw09PX.csv',index_col=0)
    

    X['class'] = y

    var = X[['designation', 'class']]

    var.designation = var.designation.astype(str)
    var.designation = var.designation.str.lower()
    var = var.drop_duplicates(subset = "designation")
    

    symbols = [",", "'",  "/", "{", "}", "[", "]","(",")", ":", ";", "°", "-", "_", "ø"]
    

    for i in symbols :
        try :
            var.replace(i, " ", regex = True, inplace = True)
        except: 
            pass
        
    
    var.replace(".", "",  inplace = True)
        
    st.subheader("Affichage de nuage de mots par type de code de produits")
    pred = st.selectbox('Sélectionner la catégorie du produit :', key = "général",options = ['10 :  Livres type romain, Couvertures de livres ', '2280 :  Livres, journaux et revues anciennes','2403 :  Livres, BD et revues de collection', '2705 :  Livres en général','2522 :  Cahiers, carnets, marque pages','40 :  Jeux videos, CDs + mais aussi equipements, cables, etc. ', '50 :  Equipements/complements consoles, gamers ', '2905 :  Jeux vidéos pour PC', '2462 :  Equipement jeux, jeux video, play stations', '60 :  Consoles ','1280 :  Jouets pour enfants, poupées nounours, equipements enfants', '1281 :  Jeux socitété pour enfants, Boites et autres, couleurs flashy', '1300 :  Jeux techniques, Voitures/drones télécomandés, Equipement, petites machines ', '1180 :  Figurines et boites ', '1140 :  Figurines, Personnages et objets, parfois dans des boites ', '1160 :  Cartes collectionables, Rectangles, beaucoup de couleurs','1320 : Matériel et meubles bébé poussettes, habits', '1560 : Meubles, matelas canapés lampes, chaises',  '2582 :  Matériel, meubles et outils pour le jardin', '2583 :  Equipements technique pour la maison et exterieur (piscines), produits', '2585 :  Idem 2583 :   Equipements technique pour la maison et exterieur (piscines), produits', '1302 :  Equipements, Habits, outils, jouets, objets sur fond blanc', '2220 :  Equipements divers pour animaux',  '1920 : Oreillers, coussins, draps', '2060 : Décorations',  '1301 : Chaussetes bébés, petites photos ', '1940 : Alimentations, conserves boites d gateaux'])     
    prediction = pred.split(' :')[0]
    
    choix_classe = prediction
    
    if st.button('valider') :
        var10 = var[var['class'] == np.int(choix_classe)]
       
        token = var10.designation.apply(word_tokenize)
       
    
        stop_words_french = set(stopwords.words("french"))
        stop_words_english = set(stopwords.words("english"))
        stop_words_german = set(stopwords.words("german"))
    
        stop_words = stop_words_french.union(stop_words_english) 
        stop_words = stop_words.union(stop_words_german )
        def stop_words_filetring(mots) : 
            tokens = []
            for mot in mots:
                if mot not in stop_words:
                    tokens.append(mot)
                return tokens
        token = token.apply(stop_words_filetring)    
        
        list_ponctuation = []
        for i in string.punctuation :
            list_ponctuation.append(i)
            
        def stop_ponctuations(mots) : 
            tokens = []
            for mot in mots:
                if mot not in list_ponctuation:
                    tokens.append(mot)
            return tokens
        token = token.apply(stop_ponctuations)
        
        def strip_accents(text):
            try:
                text = unicode(text, 'utf-8')
            except NameError: 
                    pass
            text = unicodedata.normalize('NFD', text)\
               .encode('ascii', 'ignore')\
               .decode("utf-8")
    
            return str(text)  
        
        
        
        liste_i = []
        liste_a= []
    
        for i in token :
            liste_j = []
            for j in i :
                liste_j.append(strip_accents(j))
            liste_a.append(liste_j)
            
        token=pd.Series(liste_a, index = token.index)
        
        tokene = pd.DataFrame(token, columns = ['designation'])
    
        tokenizer = RegexpTokenizer(pattern = "\w{4,}")
    
        tokene.designation = tokene.designation.astype(str)
        tokene.designation = tokene.designation.apply(tokenizer.tokenize)
        
        liste_a= []
        stemmer = FrenchStemmer()
    
        ess = pd.Series(tokene.designation)
        for i in ess :
            liste_j =[]
            for j in i:
                liste_j.append(stemmer.stem(word = j))
            liste_a.append(liste_j)
        tokene = pd.Series(liste_a, index = token.index)
    
        porter = PorterStemmer()
        lancaster=LancasterStemmer()
    
    
        liste_a= []
    
        for i in tokene :
            liste_j =[]
            for j in i:
                liste_j.append(porter.stem(word = j))
            liste_a.append(liste_j)
        tokene = pd.Series(liste_a, index = token.index)
    
        liste_a= []
    
        for i in tokene :
            liste_j =[]
            for j in i:
                liste_j.append(lancaster.stem(word = j))
            liste_a.append(liste_j)
        tokene = pd.Series(liste_a, index = token.index)
    
        wnl = WordNetLemmatizer()
    
        liste_a= []
    
        for i in tokene :
            liste_j =[]
            for j in i:
                liste_j.append(wnl.lemmatize(word = j, pos = "n"))
            liste_a.append(liste_j)
        tokene = pd.Series(liste_a, index = token.index)
        tokene = pd.DataFrame(tokene, columns = ["designation"])
    
        tokene.designation=[" ".join(text) for text in tokene.designation.values]
    
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(tokene.designation)
    
        print(Counter(vectorizer.vocabulary_))
        
        freq = Counter(vectorizer.vocabulary_)
        plt.figure(figsize = (20,10))
        plt.bar(x = pd.DataFrame(freq.most_common(15), columns = ['word', "Count"]).word.tolist(), height = pd.DataFrame(freq.most_common(15), columns = ['word', "Count"]).Count.tolist())
        
        from wordcloud import WordCloud
        wc = WordCloud(background_color = "black", max_words = 400,  height = 100, stopwords = stop_words)
    
        plt.figure(figsize= (50,50)) # Initialisation d'une figure
        wc.generate_from_frequencies(vectorizer.vocabulary_)           # "Calcul" du wordcloud
        plt.imshow(wc) # Affichage
        plt.title("Nuage de  mots de la classe :")
        plt.show()
        st.subheader('Nuage de  mots de la classe: '+ str(choix_classe))
        st.pyplot()
        

    

    
    
    #get current working directory
    current_path = os.getcwd() 
    
    #Training images path
    images_path = current_path + r'/images/image_train/'
    
    #List with the name of all training images
    images_list = os.listdir(images_path)
    

    #Create a column with the name of the picture
    X['image name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
    

    X["class"] = y
    X["class"] = X["class"].astype('str')
    
    st.title('')
    st.title('')

    
    st.subheader("Affichage du générateur d'images")

    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_data_generator = ImageDataGenerator(rescale = 1./255,  
            preprocessing_function = None,
                                        rotation_range = 10,
                                        width_shift_range = 0.1,
                                        height_shift_range = 0.1,
                                        zoom_range = 0.1,
                                        brightness_range=[0.9, 1.1],
                                        horizontal_flip = True)
    
    original = ImageDataGenerator( rescale = 1./255, 
            preprocessing_function = None  )
    
    
    # In[68]:
    
    
    batch_size = 32
    path = os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\ds_images\\images\\image_train')
    
    import streamlit as st
    pdt = st.number_input("NUMERO DE L'IMAGE",0)
    
    
    train_generator = train_data_generator.flow_from_dataframe(dataframe=pd.DataFrame(X.iloc[np.int(pdt),:]).T,
                                                              directory=path,
                                                               x_col = "image name",
                                                               y_col = "class",
                                                               class_mode ="sparse",
                                                              target_size = (240, 240), 
                                                              batch_size = batch_size, random_state = 1)
    
    
    or_generator = original.flow_from_dataframe(dataframe=pd.DataFrame(X.iloc[np.int(pdt),:]).T,
                                                              directory=path,
                                                               x_col = "image name",
                                                               y_col = "class",
                                                               class_mode ="sparse",
                                                              target_size = (240, 240), 
                                                              batch_size = batch_size, random_state = 1)
    
    
    # In[69]:
    
    
    def inversegetSamplesFromDataGen(resultData,):
        x = resultData.next() #fetch the first batch
        a =x[0]
        return a
    
    def getSamplesFromDataGen(resultData):
        x = resultData.next() #fetch the first batch
        a = x[0] # train data
        c = x[1]
        plt.imshow(a[0])
        plt.title("TRANSFORMATION")
        plt.show() 
    
    if st.button('valider', key = "2") :
        st.markdown("Type de code produit : " + str(y.iloc[np.int(pdt)][0]))
        plt.figure(figsize = (15,15))
        plt.subplot(1,2,1)
        b = inversegetSamplesFromDataGen(or_generator)
        plt.imshow(b[0])
        plt.title('ORIGINALE')
        plt.subplot(1,2,2)
        getSamplesFromDataGen(train_generator)
        st.pyplot()

                    


if navigation == "Méthode" :
    os.chdir('C:\\Users\\barry\\OneDrive - CSTBGroup\\ds_images\\Demo')
    from rakuten_demo import *

    cat = CatModel()

    st.title('Méthode')
    st.title('')
    st.title('')

    st.write('''
### Afin de résoudre le problème de classification plusieurs étapes ont été réalisées.
             
### Tout d'abord les différents types de données ont été traitées à partir de modèles de type :
### - ConvNet, pour les données images
### - RNN et Machine Learning, pour les données textes.
            
### Ensuite, après plusieurs tentatives de paramétrisation et d'hyperparamétrisation, les meilleures modèles et versions de modèle ont été intégrés dans un modèle de concaténation (cf. Graph 1 et Graph 2).
             
### Remarque : Le modèle de type concaténé présente l'avantage d'intégrer à la fois les données images et données textes pour prédire une même cible finale. Ce qui permet d'enrichir le modèle et d'améliorer les résultats.
             ''')

    st.title('')
    tf.keras.utils.plot_model(cat.model, show_shapes = True, show_layer_names = True)
    st.subheader("Graph 1 : Architecture du modèle - version manuelle")
    image = Image.open('C:/Users/barry/OneDrive - CSTBGroup/ds_images/mod_bleu_conc.png')
    st.image(image, 300);
    
    st.title('')
    tf.keras.utils.plot_model(cat.model, show_shapes = True, show_layer_names = True)
    st.subheader("Graph 2 : Architecture du modèle - version logiciel")
    image = Image.open('C:/Users/barry/OneDrive - CSTBGroup/ds_images/Demo/architecture.png')
    st.image(image, 300);
    

if navigation == "Démonstration":
    os.chdir('C:/Users/barry/OneDrive - CSTBGroup/ds_images/Demo')
    from rakuten_demo import *

    create_demo() 
    
   
    st.title('Démonstration du modèle concatenate')
    st.title('')
    st.title('')

    cat = CatModel()

    
    df = d_get_X_df()
    print("Shape de X =", df.shape)
    
    
    col1,  col2 = st.columns(2)
    
        
    ys = d_get_y()
    
    X_text = d_get_X_text()         # designation + description
    X_text_embed_multilingual = d_get_X_text_embed_multilingual()
    X_text_spacy_lemma = d_get_X_text_spacy_lemma()
    

    nb = df.shape[0] # Nombre d'échantillons
    #print("Choisir un identifiant de produit compris entre 0 et  " + str(nb))
    st.title('')
    st.title('')
    idx = st.number_input("Choisir un identifiant de produit compris entre 0 et  " + str(nb),0)
    idx = int(idx)
    # st.write(f"Classe {ys[idx]}")
    fichier_image = d_get_image(df.iloc[idx])
    image = plt.imread(fichier_image)
    with col1 :
        st.subheader('Image')
    
        plt.imshow(image)
        plt.show()
        st.pyplot()
    
    with col2 : 
        st.subheader('Texte')
    
        #st.write("============== texte brut")
        st.write(X_text[idx])
        #st.write("============== texte lemmatizé")
        #st.write(' '.join(X_text_spacy_lemma[idx]))
    
    
    
    nb = df.shape[0] # Nombre d'échantillons
    y_real = [ys[i] for i in range(nb)] # Classes réelles
    y_pred = d_predict(idx) # Prédiction des échantillons
    st.subheader('')
    st.subheader('Résultat de la prédiction :')
    st.subheader("Classe " + y_pred[0] + " :" +classes[y_pred[0]])
    st.subheader("")
    if y_pred[0] == y_real[idx] :
        st.success("   Le résultat est correct.") 
    else :     
        st.error("Le résultat n'est pas correct.")
        st.info("La véritable classe est " + y_real[idx] + " :" +classes[y_real[idx]] + ".")

    
    



if navigation == "Jeu" :
    os.chdir('C:/Users/barry/OneDrive - CSTBGroup/ds_images')
    path = os.getcwd() + '/images/image_train'
    
    y_organised = ['10','2280','2403','2705','2522',
               '40','50','2905','2462','60',
               '1280','1281','1300','1180','1140','1160',
               '1320','1560',
               '2582','2583','2585','1302','2220',
               '1920','2060',
               '1301','1940'
              ]


    X = pd.read_csv('X_train_update.csv',index_col=0)
    y = pd.read_csv('Y_train_CVw08PX.csv',index_col=0).squeeze().map(str)

    X = X.merge(y, right_index = True, left_index = True)


    df = pd.read_csv("C:/Users/barry/OneDrive - CSTBGroup/ds_images/Demo/pred_model.csv")
    X = X.merge(df, on = "productid", how = "inner")


    X = X[['designation', 'description', 'productid', 'imageid', 'prdtypecode' , 'y_pred']]
    X.columns = ['designation', 'description', 'productid', 'imageid', 'prdtypecode' , 'y_pred']

    y = X.prdtypecode

    y_model = X.y_pred.tolist()

    X.drop('y_pred', axis = 1, inplace = True)
    X.drop('prdtypecode', axis = 1, inplace = True)

#Create a column with the name of the picture
    X['image_name'] = 'image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'
    X['image_path'] = path + r'/image_' + X['imageid'].map(str) + '_product_' + X['productid'].map(str) + '.jpg'



    def show_image(i):
        
        img = cv2.imread(path + '/' + X.iloc[i]['image_name'])
        img = cv2.resize(img, (560, 560), interpolation=cv2.INTER_CUBIC)
    
        plt.figure(figsize = (6,6))
        plt.subplot(1,1,1)
    
    
        plt.axis('off')
        plt.imshow(img, cmap=cm.binary, interpolation='None')
        st.pyplot()
        #plt.title("Classifiez l'image suivante");



    def show_text(i):
        st.write('Designation :',X['designation'].iloc[i])
        st.write('Description :',X['description'].iloc[i])
        
        
        
    def show_concat(i ):
        col1,  col2 = st.columns(2)
        img = cv2.imread(path + '/' + X.iloc[i]['image_name'])
        img = cv2.resize(img, (560, 560), interpolation=cv2.INTER_CUBIC)
    
        with col1 :
            plt.figure(figsize = (6,6))
            plt.subplot(1,1,1)
    
    
            plt.axis('off')
            plt.imshow(img, cmap=cm.binary, interpolation='None')
            st.pyplot()
        with col2 :
            st.write('Designation :',X['designation'].iloc[i])
            st.write('Description :',X['description'].iloc[i])
    
    st.markdown(" # **Jeu **")
    st.text('')
    st.markdown(" ## Classification manuelle des produits Rakuten")
    st.text('')
    st.text('')
    
    st.markdown('')
    st.markdown('')
    st.markdown("## Le but du jeu consiste à classer manuellement les différents produits présents dans la base de données.") 
    st.text('')
    
    st.markdown("### Chacun de ces produits est présenté à partir des données : ")
    st.markdown('* ###  Textes et Images')
                
    st.text('')
    
    st.markdown("###  Parviendrez-vous à battre notre modèle concatenate ?")
    st.text('')
    
    st.text('')    
    
    def show_product(i = st.number_input("Entrer un identifiant de produit compris entre 0 et 539 :", 0)):
    
        
        show_concat(i)
        type_pred = 'concat'
    
        return i, type_pred
    
    
    man_cla = pd.DataFrame(columns = ['y_reel','y_pred', 'type', 'idx_prod'])
    def classification_manuelle(i,type_pred, prediction):
        
        st.text('')
    
        if prediction == y[i]:
            st.success('VOTRE PREDICTION EST CORRECTE !')
        else:
            st.error('VOTRE PREDICTION EST FAUSSE !' )
    
        st.text('')
        
        if str(y_model[i]) == str(y[i]) :
            st.success('LE MODELE A BON !')
        else :
            #st.error('LE MODELE SE TROMPE !' )
            st.error('LE MODELE SE TROMPE !    Il sélectionne la classe : ' + str(y_model[i]) + '-'  + classes[str(y_model[i])])
            
                    
        st.subheader('La vraie classe est: ' + y[i] + ' - ' + classes[y[i]]  )
    
        new_man_cla = man_cla.append({'y_reel':y[i],'y_pred':prediction,'type':type_pred,'idx_prod':str(i)},ignore_index=True)
        return prediction, new_man_cla
    
    
    def conf_matx(y_test,y_pred):
        cnf_matrix = metrics.confusion_matrix(y_test,y_pred,labels=y_organised)
    
        pond_matrix = []
        for line in cnf_matrix:
            pond_line = []
            for cell in line:
                pond_line.append(round(cell/sum(line),2))
            pond_matrix.append(pond_line)
            #print(sum(line))
            #print(sum(pond_line))
        cnf_matrix = np.array(pond_matrix)
        ###Optionnel: Afficher une matrice de confusion sous forme de tableau coloré
        #classes = set(y_pred)
        classes = y_organised
        return cnf_matrix
    
    
    
    #Explo
    #####
    
    
    ### st.session_state
    
    
    from sklearn.metrics import classification_report
    #a, b , c ,d= show_product()
    #♣show_product(d,c)
    #JEU
    #####
    
    
        
    if navigation == "Jeu" :
        with st.form("S1"):
            i,type_pred = show_product()
            if str(i) not in st.session_state :
                st.session_state[str(i)] = i
                
    
            pred = st.selectbox('Sélectionner la catégorie du produit :', key = "choix",options = ['***Livres***',  '10 :  Livres type romain, Couvertures de livres ', '2280 :  Livres, journaux et revues anciennes','2403 :  Livres, BD et revues de collection', '2705 :  Livres en général','2522 :  Cahiers, carnets, marque pages','***Jeux***', '40 :  Jeux videos, CDs + mais aussi equipements, cables, etc. ', '50 :  Equipements/complements consoles, gamers ', '2905 :  Jeux vidéos pour PC', '2462 :  Equipement jeux, jeux video, play stations', '60 :  Consoles ','***Jouets et Figurines***', '1280 :  Jouets pour enfants, poupées nounours, equipements enfants', '1281 :  Jeux socitété pour enfants, Boites et autres, couleurs flashy', '1300 :  Jeux techniques, Voitures/drones télécomandés, Equipement, petites machines ', '1180 :  Figurines et boites ', '1140 :  Figurines, Personnages et objets, parfois dans des boites ', '1160 :  Cartes collectionables, Rectangles, beaucoup de couleurs','***Meubles***', '1320 : Matériel et meubles bébé poussettes, habits', '1560 : Meubles, matelas canapés lampes, chaises', '***Equipements***', '2582 :  Matériel, meubles et outils pour le jardin', '2583 :  Equipements technique pour la maison et exterieur (piscines), produits', '2585 :  Idem 2583 :   Equipements technique pour la maison et exterieur (piscines), produits', '1302 :  Equipements, Habits, outils, jouets, objets sur fond blanc', '2220 :  Equipements divers pour animaux',  '***Décorations***', '1920 : Oreillers, coussins, draps', '2060 : Décorations', '***Autre***', '1301 : Chaussetes bébés, petites photos ', '1940 : Alimentations, conserves boites d gateaux'])     
            
            if pred not in ['***Livres***', '***Jeux***', '***Jouets et Figurines***'] :
    
                prediction = pred.split(' :')[0]
                if st.form_submit_button('valider') :
     
                    prediction, new_man_cla = classification_manuelle(i,type_pred, prediction)
                    
    

            
            else :
                if st.form_submit_button('valider') :
                    st.error('Error message')

if navigation == "Bilan" :
    st.title("Bilan")
    st.title("")
    st.title('')
    st.markdown(''' 
### Les résultats de scoring f1 weighted  obtenus en interne, à la date de diffusion du rapport sont de **0.8783**.

### Les prédictions soumises le 9 août 2021 sur le site du challenge ont permis à notre FEEEScientest d'atteindre un score de **0.8628**.

###  L'équipe est par ailleurs classée **9ème**.

''')
    st.title('')
    st.header("Classement - site du challenge")
    st.subheader('')
    image = Image.open('C:/Users/barry/OneDrive - CSTBGroup/ds_images/classement.png')
    st.image(image, 300);
    