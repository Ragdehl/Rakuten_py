# -*- coding: utf-8 -*-
"""

"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

import spacy
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

@st.cache(suppress_st_warning=True)
def read_csv(file):
    return pd.read_csv(file)

def read_X(X_file):
    """
    Lecture d'un fichier de données X et petit nettoyage

    """
    df = read_csv(X_file)
    # La première colonne est un doublon:
    #          df["Unnamed: 0"] == df.index 
    df = df.drop("Unnamed: 0", axis=1)
    # Les colonnes designation et description sont à l'origine de type objet
    #          df.info()
    df.designation = df.designation.astype('string')
    df.description = df.description.astype('string')
    return df

def image_path(row, subdir="image_train"):
    """
    Construction du chemin d'un fichier image
    """
    f = "image_%d_product_%d.jpg" % (row.imageid, row.productid) 
    ff = os.path.join(os.getcwd(), "images", subdir, f)
    return ff if os.path.isfile(ff) else None
    
def X_image_paths(X, n):
    """
    Construction de la séries des paths des images avec utilisation d'un
    cache sur le disque
    """
    f = os.path.join(os.getcwd(), n + ".pkl")
    t = os.path.join(os.getcwd(), n + ".tmp")
    if os.path.exists(t):
        os.remove(t)
    if os.path.isfile(f):
        return pickle.load(open(f, "rb"))
    with st.spinner("Construction et vérification des fichiers images..."):
        imagepaths = X.iloc[:,:].apply(image_path,  axis=1)
        nbnone = imagepaths[imagepaths == None].sum()
        if nbnone == 0:
            st.write(f"Les {len(imagepaths)} fichiers images sont valides")
            
        else:
            st.error(f"{nbnone} fichiers non trouvés sur {len(imagepaths)}")
        pickle.dump(imagepaths, open(t, "wb"))
        os.rename(t, f)
        return imagepaths

def X_image_attributes(imagepaths, n):
    """
    Construction d'un dataframe avec des attibuts des images, en utilisant
    un cache sur disque
    """
    f = os.path.join(os.getcwd(), n + ".pkl")
    t = os.path.join(os.getcwd(), n + ".tmp")
    if os.path.exists(t):
        os.remove(t)
    if os.path.isfile(f):
        return pd.read_pickle(f)
    with st.spinner(f"Construction de {n} ..."):
        progress = st.progress(0)
        oneperc = len(imagepaths) // 100
        data = {"imgstsize" : []}
        for i, p in enumerate(imagepaths):
            data["imgstsize"].append(os.stat(p).st_size)
            if i % oneperc == 0:
                perc = (100 * i) // len(imagepaths)
                progress.progress(perc + 1)
        df = pd.DataFrame(data, index=range(len(imagepaths)))
        df.to_pickle(t)
        os.rename(t, f)
        return df

def X_text(X):
    """
    Liste de textes correpondant à un dataframe
    """
    lst = []
    for desc, desi in zip(X.description, X.designation):
        s = desc if type(desc) == str else ''
        s+= desi if type(desi) == str else ''
        lst.append(s)
    return lst
    
def tokenize_spacy(sentence, nlp=None):
    """
    Tokenizer basé sur spacy
    """
    if nlp is None:
        nlp = spacy.load("fr_core_news_sm")
        nlp.disable_pipes ('tagger', 'parser', 'ner')
    return [x.text for x in nlp(sentence.lower())]

def predict_sklearn_vc(X_trn, X_tst, y_trn, y_tst):
    """
    Prédiction avec sklearn.TfidfVectorizer
    """
    
    tv = TfidfVectorizer(analyzer='word',
#                  tokenizer=pronto_text_tokenizer_kneigh,
                  strip_accents='ascii',
                  #stop_words=list(stopwords.words('french')),
                  max_df=0.8,
                  min_df=1,
                  ngram_range=(1,1),
                  use_idf=False,
                  smooth_idf=False,
                  sublinear_tf=False,
                  binary=False,
                  #max_features=10000,
                  )
    X_trn_vect = tv.fit_transform(X_trn)
    X_tst_vect = tv.transform(X_tst)
    knn = KNeighborsClassifier()
    knn.fit(X_trn_vect, y_trn)
    return knn.predict(X_tst_vect)
  
if __name__ == '__main__':
    
    st.set_page_config(layout='centered')
    #pd.set_option("display.max_rows", 500)
    #nltk.download('stopwords')
    
    st.title("RAKUTEN_PY")
    
    #
    # Résumé du projet
    #
    st.header("1. Résumé du projet")
    st.markdown("Rakuten France veut pouvoir catégoriser ses produits" 
                " automatiquement grâce à la désignation, la description"
                " et l'image des produits vendus sur leur site.")
    st.markdown("Les données suivantes ont été fournies dans le cadre d'un"
             " concours que Raduken a lancé afin de sélectionner le"
             " meilleur modèle de classification de ses produits:")
    st.markdown("  * **X_train**: donnéees d'entrainement contenant,  \n"
                "       - designation: texte désignant le produit  \n"
                "       - description: texte décrivant le produit  \n"
                "       - productid: identifiant numérqiue du produit  \n"
                "       - imageid: identifiant numérique de l'image  \n"
                "  *  **X_test**: données de même structure que X_train"
                " qui seront utilisées par Rakuten pour tester le modèle  \n"
                "  * **Y_train**: contient l'identifiant de la catégorie"
                " du produit  \n"
                "  * **images**: un dossier contenant un fichier image par"
                " produit")
    st.image(plt.imread("Rakuten1.png"))
    
    #
    # Vision du projet
    #
    st.header("2. Vision du projet")
    st.write("TODO: inclure img3 + texte à voir...")
    
    #
    # Lecture et petit toilettage des données
    #
    X_train = read_X("X_train_update.csv")
    y_train = pd.read_csv(
        "Y_train_CVw08PX.csv").drop("Unnamed: 0", axis=1)["prdtypecode"]
    
    #
    # Exploration des données brutes
    #
    st.header("3. Exploration des données d'entrainement brutes")
    #
    if st.checkbox("Exploration aléatoire des échantillons"):
        if st.button("Visualiser un échantillon et quelques autres de même type"):
            df = pd.concat([X_train, y_train], axis=1)
            idx = int(np.random.random_sample() * X_train.shape[0])
            row = X_train.iloc[idx]
            st.markdown("**designation**")
            st.text(row.designation)
            st.markdown("**description**")
            st.text(row.description)
            f = image_path(X_train.iloc[idx],"image_train")
            if f is None:
                st.write("Pas de fichier image")
            else:
                img = plt.imread(f)
                col = st.beta_columns(1)
                col[0].header("index %d - prdtypecode %d" % (idx, y_train[idx]))
                col[0].image(img)
            prdtypecode = y_train[idx]
            index = y_train[y_train == prdtypecode].index.drop(idx)
            randi = (np.random.random_sample(4)*len(index)).astype(int)[:4]
            idxlst = [index[i] for i in randi]
            cols = st.beta_columns(len(idxlst))
            for i, idx in enumerate(idxlst):
                f = image_path(X_train.iloc[idx],"image_train")
                if not f is None:
                    img = plt.imread(f)
                    cols[i].header("index %d" % idx)
                    cols[i].image(img)
    #
    st.markdown("La table X_train est composée de **84916 échantillons**.  \n  \n"
                "Il faut ôter la première colonne de X_train, qui est une"
                " duplication de l'index, et convertir les colonnes 'designation'"
                " et 'description' en type 'string', plus adapté.\n")
    st.write(X_train.iloc[:1000,:])
    st.markdown("Les variables 'designation' et 'description' contiennent"
                " des valeurs non uniques (des valeurs que l'on retouvent"
                " dans plusieurs échantillons). ces variables contiennent"
                " beaucoup de termes redondants."
                " La variable 'description' contient plus d'un tier de"
                " valeurs manquantes.")
    st.write(pd.DataFrame({
        "Valeurs manquantes": X_train.isna().sum(),
        "Valeurs non uniques": X_train.apply(lambda x: x.duplicated(keep=False).sum(),
                                           axis=0)}))
    st.markdown("Il y a des designations récurrentes pour des produits"
                " enregistrés sous des Id différents. La désignation la plus"
                " fréquente se retrouve 76 fois.")
    with st.spinner("Comptage des valeurs de 'designation' ..."):
        st.write(X_train.designation.value_counts()[:2000])
    st.markdown("Il y a des descriptions récurrentes pour des produits"
                " enregistrés sous des Id différents. La description la plus"
                " fréquente se retrouve 252 fois.")
    with st.spinner("Comptage des valeurs de 'description' ..."):
        st.write(X_train.description.value_counts()[:2000])
    f1 = "[weighted f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#:~:text=The%20F1%20score%20can%20be%20interpreted%20as%20a,%2A%20%28precision%20%2A%20recall%29%20%2F%20%28precision%20%2B%20recall%29)"
    st.markdown("La série y_train inclut **27 catégories**; les populations"
                " de ces catégories sont déséquilibrées en taille, ce"
                " qui devra être pris en compte pour la classification. Cela"
                " explique sans doute le choix de la métrique " f"{f1}"
                " par Rakuten.")
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(15,4))
    sns.countplot(x=y_train, facecolor=(0,0,0,0), linewidth=5,
                  edgecolor=sns.color_palette("dark", 3))
    plt.title("Répartition des échantillons dans chacune des catégories")
    #          fontdict = {'fontsize' : 25})
    #plt.xticks(fontsize=15)
    #plt.xlabel("prdtypecode", fontsize=25)
    plt.ylabel("nombre d'échantillons")
    #plt.yticks(fontsize=15)
    st.pyplot(fig)
    #
    # Exploration des données textuelles
    #
    st.header("4. Exploration des données textuelles")
    #
    if st.checkbox("Exploration aléatoire du texte"):
        if st.button("Tokenizer et visualiser des mots extraits du texte"):
            randlst = list(np.random.random_sample(500) * X_train.shape[0])
            d = X_train.iloc[[int(r) for r in randlst]]
            # python -m spacy download fr_core_news_sm (ou lg..)
            nlp = spacy.load("fr_core_news_sm")
            nlp.disable_pipes ('tagger', 'parser', 'ner')
            descriptions = list(d.description.unique())
            designations = list(d.designation.unique())
            ls = [x for x in descriptions + designations if type(x) == str ]
            ls = list(set(ls))
            st.write(f"Extraction du vocabulaire à partir de {len(ls)} phrases")
            mots = list()
            progress = st.progress(0)
            oneperc = len(ls)//100
            for i, s in enumerate(ls):
                mots += tokenize_spacy(s, nlp)
                if i % oneperc == 0:
                    perc = (100 * i) // len(ls)
                    progress.progress(perc + 1)
            mots = list(set(mots))
            st.write(f"Taille du vocabulaire = {len(mots)}")
            st.write(list(set(mots)))
    #
    st.markdown("Le tokenizer français de spacy semble bien reconaitre"
                " les mots. Un nettoyage est nécessaire cependant:  \n"
                "  * des morceaux de balises htlml polluent le texte  \n"
                "  * il semble judicieux de tout transformer en minuscule  \n"
                "  * il y a des caractères spéciaux isolés qui trainent  \n"
                "  * les mots vides (stop words) sont à retirer  \n"
                "  * etc.")
    #
    # Predictions à parir du texte
    #
    st.header("5. Prédictions à partir des données textuelles")
    #
    X, y = X_text(X_train), y_train
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2)
    #
    predictor = "sklearn.TfidfVectorizer"
    if st.button(f"Prédiction avec {predictor}"):
        with st.spinner("Prédiction en cours ..."):
            y_pred = predict_sklearn_vc(X_trn, X_tst, y_trn, y_tst)
            score = f1_score(y_tst, y_pred, average='weighted')
            st.write(f"F1-score (weighted) = {score}")
 
    #
    # Exploration des images
    #
    st.header("6. Exploration des images")
    #
    st.markdown("Après avoir vérifier que la forme (shape) et la taille (size)"
                " des images étaient identiques on constate néanmoins que la"
                " taille des fichiers 'png' varie (stat.st_size) et ne semnle"
                " pas suivre une loi normale: Il serait"
                " peut être intéressant de voir si on peut utiliser cette"
                " information.")
    imgpaths = X_image_paths(X_train, "X_train_image_paths")
    imgatts = X_image_attributes(imgpaths, "X_train_image_attributes")
    st.write(imgatts.describe().T)
    fig = plt.figure(figsize=(20,5))
    plt.hist(imgatts['imgstsize'], bins=40, rwidth=0.8, alpha=0.5)
    plt.title("Répartition des effectifs en fonction des tailles de fichiers")
    plt.xlabel("tailles de fichiers")
    plt.ylabel("effectifs")
    st.pyplot(fig)

            

    #
    # Prédictions à partir des images
    #
    st.header("7. Prédictions à partir des images")
    #
    
 


    

    
        
  
      
