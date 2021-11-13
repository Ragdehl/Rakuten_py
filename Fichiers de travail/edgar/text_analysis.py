# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:08:07 2021

@author: Edgar
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize


# Initialiser la variable des mots vides
stop_words = set(stopwords.words('french'))

df_X = pd.read_csv(r'C:\Users\Edgar\Documents\Rakuten\X_train_update.csv')
df_y = pd.read_csv(r'C:\Users\Edgar\Documents\Rakuten\Y_train_CVw08PX.csv')

lemmatizer = WordNetLemmatizer()

def lemma(sentence): #Lemmatizer
    doc = word_tokenize(sentence, language='french')
    return [lemmatizer.lemmatize(token) for token in doc]

def stop_words_filetring(mots) : 
    tokens = []
    for mot in mots:
        if mot not in stop_words:
            tokens.append(mot)
    return tokens

def clean_text(text):
    tokens = []
    words = word_tokenize(text.lower(), language='french')
    for word in words:
        if word not in stop_words:
            tokens.append(lemmatizer.lemmatize(word))
    return tokens

X = df_X.designation.astype(str) + ' ' + df_X.description.astype(str)
y = df_y.prdtypecode

X_clean = X.apply(lambda cell: clean_text(cell))



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialiser un objet vectorisateur
#vectorizer = CountVectorizer()

# Mettre à jour la valeur de X_train et X_test
#X_train = vectorizer.fit_transform(X_train).todense()
#_test = vectorizer.transform(X_test).todense()
