# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:15:45 2021

@author: utilisateur
"""
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# App Simple pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')

st.sidebar.header("Les parametres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longeur du Sepal',4.3,7.9,5.3)
    sepal_width=st.sidebar.slider('La largeur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longeur du Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
    }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

df=user_input()

st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)

iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)

st.subheader("La catégorie de la fleur d'iris est:")
st.write(iris.target_names[prediction])
