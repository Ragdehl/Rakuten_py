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
from wordcloud import WordCloud
import streamlit as st
# import altair as alt

st.title('RAKUTEN Python Data Visualization')
st.markdown("""
* Application test for MASTER FINAL PRESENTATION
""")

df = pd.read_csv('X_train_update.csv',index_col=0)

designation = df['designation']
description = df['description']

### TEST LE 31 08 (boutons barre de gauche)
with st.sidebar:
            st.title("RAKUTEN PROJECT")
            st.subheader("Select a page below:")
            mode = st.radio(
                "Menu",
                [
                    "DataViz",
                    "Model Game",
                    "A little theory"
                ],
            )
    
    
        #st.sidebar.header('User Data Selection')
            st.sidebar.subheader('Variables Désignation & Description')
            st.sidebar.subheader('Target')
        
        #### TESTS KO
        # selector=st.sidebar.multiselect[description, designation]
        
        #options = st.multiselect(
        #    [description, designation]
        #st.write('You selected:', options)
if mode == "DataViz" :      
            # ETUDE DES NAN'S
            st.header('Observation des données texte manquantes')     
            figW=plt.figure() 
            sns.heatmap(df.isna(),cmap="coolwarm", center = 10.0, cbar=False) #(bins = 100, color='gray')    
            st.pyplot (figW)
            
            
            #sorted_text_data = st.sidebar.multiselect(designation, description)
            # df_selected_sector = df[df.isin(sorted_text_data)]
            st.write(designation)
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
                
                
            #df['target'] = target
            ##percent_target = 100 * df['target'].value_counts()/len(df)
            #percent_target = df['target'].value_counts(sort=True, ascending=True)
            #if st.button('Show Plot Target'):
            #    figz=plt.figure() 
            #    sns.barplot (x=df['target'], y=percent_target, color='gray') #(bins = 100, color='gray')  
            #    plt.xticks(rotation=90)
            #    st.pyplot (figz) 
    
elif mode == "Model Game":
        st.title("Model Game")
        st.write("Choose the selected label of image")
        
        # CODE EDGAR#####
        
        st.sidebar.write('selected class:')
        # Initialization
        if 'key' not in st.session_state:
            st.session_state['key'] = 'value'
        
        #st.json({'name':"Jesse",'gender':"male"})
        classes ={'Livre occasion' : 10,'Journaux et revues' :2280}
        #,2403,2522,2705,40,50,60,2462,2905,1140,1160,1180,1280,1281,
        #          1300,1302,1560,2582,1320,2220,2583,2585,1920,2060,1301,1940}
        
        for classe in classes:
            for label in classe :
                if st.session_state[label]:
                    class_man = classe [label]
                    st.sidebar.write(class_man)
                    
        # init
        if 'true_class' not in st.session_state:
            st.session_state['true_class']='value'
            
        if class_man==st.session_state['true_class']:
            st.sidebar.write('Bonne réponse')
            
        else :
            st.sidebar.write('Mauvaise réponse')
            
        st.sidebar.write('Vraie classe :')
        st.session_state['true_class']=all_classes[y[i].item()]  
                   
        
        @st.cache
        def load_tf_model():
            path=r'C:\Users\utilisateur\CatModel_84916_model.hdf5'
            saved_model = tf.keras.models.load_model(path)
            # saved_model.load_weights(model_path + filename + '.hdf5')
            return saved_model()
        
        model = load_tf_model()
        
        test_datagen=tf.keras.preprocessing.image.ImageDataGenerator
    

        
        
         
        
       
        
elif mode == "A little theory":
    
        st.title("Theory of CNN")

        st.write(" Sources : Exemple of efficient CNN pretrained model" 
                 "https://keras.io/api/applications/efficientnet/. "
                 "Exemple of API CONCATENATE OF MODELS" 
                 "https://keras.io/api/layers/merging_layers/concatenate/.")
            
        
        if st.button('CNN'):
    
        
            uploaded_file = st.file_uploader("CONV_1", type="png")
    
        if uploaded_file is not None:
        # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
        
            # Now do something with the image! For example, let's display it:
            st.image(opencv_image, channels="BGR")
        
        
        




    
    
    
    
    
    
    
