import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import cv2 # opencv
import datetime
import hashlib
import inspect
import pickle
import wget
import gzip
import time
import tqdm
import html
import sys
import os
import re

import sklearn
import spacy
import nltk
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

import tensorflow_text # Needed for sentencepiece
import tensorflow_hub as hub

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D,\
                                    Input, Embedding, GRU, Bidirectional, \
                                    Conv1D, MaxPooling1D, GlobalMaxPooling1D, \
                                    BatchNormalization, concatenate

# Tous les modèles et sous modèles doivent utiliser exactement
# le même partitionnement train / validation / test
NB_ECHANTILLONS = -1 # -1 <=> tous les échantillons
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.15 # The validation data is selected from the last samples 
                        # in the x and y data provided, before shuffling

# Données d'entrée (lecture seule)
X_TRAIN_CSV_FILE = "X_train_update.csv"
Y_TRAIN_CSV_FILE = "Y_train_CVw08PX.csv"
X_TEST_CSV_FILE = "X_test_update.csv"
NB_CLASSES = 27

# Données de travail et de sortie
OUTDIR = "modele_rakuten_out"

MULTILINGUAL_DIR = "tfhub/universal-sentence-encoder-multilingual-large-3"
MULTILINGUAL_LINK = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

def init_OUTDIR():
    """ 
    Initialise le répertoire où seront générés les fichiers.
    Les fichiers d'entrées y sont copiés après mélange, au cas où ils
    auraient un ordre particulier
    """
    xtrainfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    ytrainfile = os.path.join(OUTDIR, os.path.basename(Y_TRAIN_CSV_FILE))
    xtestfile = os.path.join(OUTDIR, os.path.basename(X_TEST_CSV_FILE))
    if os.path.isfile(xtrainfile) and os.path.isfile(ytrainfile) and \
       os.path.isfile(xtestfile):
        return
    print('Création des fichiers entrées après mélange (shuffle)')
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
        if not os.path.isdir(OUTDIR):
            raise ValueError(f"Impossible de créer {OUTDIR}")
    dfx = pd.read_csv(X_TRAIN_CSV_FILE)
    dfy = pd.read_csv(Y_TRAIN_CSV_FILE)
    dfx, dfy = sklearn.utils.shuffle(dfx, dfy, random_state=51)
    dfx, dfy = sklearn.utils.shuffle(dfx, dfy, random_state=52)
    dfx.to_csv(xtrainfile)
    dfy.to_csv(ytrainfile)
    dfx = pd.read_csv(X_TEST_CSV_FILE)
    dfx = sklearn.utils.shuffle(dfx, random_state=53)
    dfx = sklearn.utils.shuffle(dfx, random_state=54)
    dfx.to_csv(xtestfile)

def image_path(row, subdir):
        f = "image_%d_product_%d.jpg" % (row.imageid, row.productid) 
        ff = os.path.join(os.getcwd(), "images", subdir, f)
        return ff if os.path.isfile(ff) else None
    
def get_y():
    """ Retourne les y (cibles) """
    init_OUTDIR()
    ytrainfile = os.path.join(OUTDIR, os.path.basename(Y_TRAIN_CSV_FILE))
    return pd.read_csv(ytrainfile)["prdtypecode"].astype(str)

    
def get_X_text(inputXfile=None):
    """
    Retourne une liste X de phrases pré-traitées à partir d'un dataframe
    inputXfile qui contient les colonnes "designation" et "description"
    (cette dernière possèdent des valeurs NA)
    """
    if inputXfile is None:
        inputXfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    f = re.sub(r"\.csv$", "_text.pkl", os.path.basename(inputXfile))
    xtextfile = os.path.join(OUTDIR, f)
    if os.path.isfile(xtextfile):
        with open(xtextfile, 'rb') as fd:
            return pickle.load(fd)
    print(f"Creation de {xtextfile}")
    init_OUTDIR()
    X = []
    df = pd.read_csv(inputXfile)
    for desi, desc in zip(df.designation, df.description):
        desistr = desi if type(desi) == str else ''
        descstr = desc if type(desc) == str else ''
        s = (desistr + '. DESCRIPTION: ' + descstr) if len(descstr) > 0 else desistr
        X.append(s)
    d = os.path.dirname(xtextfile)
    if not os.path.isdir(d):
        os.mkdir(d)
    pickle.dump(X, open(xtextfile, 'wb'))
    return X


def get_X_image_path(inputXfile=None):
    """
    Retourne une liste X de chemins des fichiers images à partir d'un
    dataframe inputXfile
    """
    if inputXfile is None:
        inputXfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    f = re.sub(r"\.csv$", "_image_path.pkl", os.path.basename(inputXfile))
    xpathfile = os.path.join(OUTDIR, f)
    if os.path.isfile(xpathfile):
        with open(xpathfile, 'rb') as fd:
            return pickle.load(fd)
    print(f"Creation de {xpathfile}")
    init_OUTDIR()
    df = pd.read_csv(inputXfile)
    subdir = "image_train" if 'train' in inputXfile else "image_test"
    X = list(df.apply(lambda x: image_path(x, subdir), axis=1))
    d = os.path.dirname(xpathfile)
    if not os.path.isdir(d):
        os.mkdir(d)
    pickle.dump(X, open(xpathfile, 'wb'))
    return X


def get_X_text_spacy_lemma(inputXfile=None):
    """
    Retourne une liste X de tokens créés par le tokenizer.lemma de Spacy
    """
    if inputXfile is None:
        inputXfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    f = re.sub(r"\.csv$", "_text_spacy_lemma.pkl",
               os.path.basename(inputXfile))
    xtokenfile = os.path.join(OUTDIR, f)
    if os.path.isfile(xtokenfile):
        with open(xtokenfile, 'rb') as fd:
            return pickle.load(fd)
    X_text = get_X_text(inputXfile)
    print(f"Tokenization de {len(X_text)} phrases")
    spacynlp = spacy.load("fr_core_news_sm")
    spacynlp.disable_pipes('tagger', 'parser', 'ner')
    X = []
    for sentence in tqdm.tqdm(X_text):
        sentence = re.sub(r"([?¿.!,:;])", r" \1 ", sentence) # Isole la ponctuation
        tokens = [x.lemma_ for x in spacynlp(sentence)]
        X.append(tokens)
    #X = np.array(X).reshape(len(X_text),-1)
    d = os.path.dirname(xtokenfile)
    if not os.path.isdir(d):
        os.mkdir(d)
    pickle.dump(X, open(xtokenfile, 'wb'))
    return X


def get_X_text_spacy_lemma_lower(inputXfile=None):
    """
    Retourne une liste X de tokens créés par le tokenizer.lemma de Spacy
    et transformation en caractètes minuscules
    """
    if inputXfile is None:
        inputXfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    f = re.sub(r"\.csv$", "_text_spacy_lemma_lower.pkl",
               os.path.basename(inputXfile))
    xtokenfile = os.path.join(OUTDIR, f)
    if os.path.isfile(xtokenfile):
        with open(xtokenfile, 'rb') as fd:
            return pickle.load(fd)
    X_raw = get_X_text_spacy_lemma(inputXfile)
    print(f"Mise en minuscule de {len(X_raw)} listes de tokens")
    X = []
    for sentence in tqdm.tqdm(X_raw):
        tokens = [x.lower() for x in sentence]
        X.append(tokens)
    d = os.path.dirname(xtokenfile)
    if not os.path.isdir(d):
        os.mkdir(d)
    pickle.dump(X, open(xtokenfile, 'wb'))
    return X


def get_X_text_embed_multilingual(inputXfile=None):
    """
    Retourne une liste X de vecteurs de plongement dans le modèle pre-entrainé
    "multilingual-large-3"
    """
    if not os.path.isdir(MULTILINGUAL_DIR):
        raise ValueError(f"Répertoire non trouvé {MULTILINGUAL_DIR}")
        
    if inputXfile is None:
        inputXfile = os.path.join(OUTDIR, os.path.basename(X_TRAIN_CSV_FILE))
    f = re.sub(r"\.csv$", "_text_embed_multilingual.pkl",
               os.path.basename(inputXfile))
    xembedfile = os.path.join(OUTDIR, f)
    if os.path.isfile(xembedfile):
        with open(xembedfile, 'rb') as fd:
            return pickle.load(fd)
    X_text = get_X_text(inputXfile)
    print(f"Chargement de {MULTILINGUAL_DIR}")
    embed = hub.load(MULTILINGUAL_DIR)
    print(f"Plongement de {len(X_text)} phrases")
    X = []
    for x in tqdm.tqdm(X_text):
        X.append(embed(x)) # phrase x traduite en vecteur embedding
    X = np.array(X).reshape(len(X_text),-1)
    d = os.path.dirname(xembedfile)
    if not os.path.isdir(d):
        os.mkdir(d)
    pickle.dump(X, open(xembedfile, 'wb'))
    return X


def plot_history(title, history):
    """
    Affiche les évolution par epoque de la perte et de l'accuracy
    """
    if len(history.history['loss']) <= 1:
        return
    plt.figure(figsize=(15,3))
    for i, s in enumerate(['loss', 'accuracy']):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[s], 'bo', color='orange')
        plt.plot(history.history['val_' + s], 'b', color='blue')
        plt.title(f'{title}: {s} by epoch')
        plt.ylabel(s)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='right')
    plt.show()


class RakutenBaseModel:
    """
    Classe de base héritée par les modèles
    """

    def __init__(self, name, nb=None):
            
        self.outdir = OUTDIR
        self.name = name
        self.batch_size = 32
        self.fbestweights = os.path.join(self.outdir,
                                         self.name + '_bestweights.hdf5')
        if nb is not None:
            fprefix = os.path.join(self.outdir, f"{self.name}_{nb}")
            objectfile = fprefix + "_object.pkl"
            modelfile = fprefix + "_model.hdf5"
            if not os.path.isfile(objectfile):
                raise ValueError(f"Pas de fichier {objectfile}")
            print(f"Chargement de l'objet ({objectfile})")
            obj = pickle.load(open(objectfile, "rb"))
            objvars = vars(obj)
            for v in objvars:
                setattr(self, v, objvars[v])
            if os.path.isfile(modelfile):
                print(f"Chargement du modèle ({modelfile})")
                self.model = tf.keras.models.load_model(modelfile)
        return self

    def prt(self, msg):
        """ Imprime un message et met à jour la variable 'journal' """
        if not hasattr(self, 'journal'):
            self.journal = ''
        now = datetime.datetime.now().strftime("%Hh%Mmn")
        s = f"++ [{now}] {self.name}: {msg}"
        self.journal += s + '\n'
        print(s)

    def report(self, y_test=None, y_pred=None):
        """ Affiche un rapport """
        score = round(f1_score(y_test, y_pred, average='weighted'), 4)
        self.prt(f'w-f1-score = \033[1m{score}\033[0m\n')
        print(classification_report(y_test, y_pred))
        return score

    def get_model(self):
        inp, x = self.get_model_body()
        x = tf.keras.layers.Dense(NB_CLASSES, activation='softmax', name='dense_' + self.name)(x)
        return Model(inp, x)

    def preprocess_y_train(self, off_start, off_end, input_file=None):
        y_train = get_y()[off_start:off_end]
        self.fit_labels = {i: v for i, v in enumerate(sorted(list(set(y_train))))}
        assert len(self.fit_labels) == NB_CLASSES
        rv = {self.fit_labels[i]: i for i in self.fit_labels}
        y_train = np.array([rv[v] for v in y_train])
        return y_train

    def layer_name(self, s):
        self.layer_index += 1
        return f"{s}_{self.layer_index}_{self.name}"

    def compile_and_train_gen(self, X_train, y_train, X_val, y_val,
                              optimizer='adam',
                              epochs=10,
                              patience_stop=2, patience_lr=None,
                              class_weight=[],
                              callbacks=[]):
        self.prt("fit(): Début")
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_stop,
                                             restore_best_weights=True, verbose=1),
                     tf.keras.callbacks.ModelCheckpoint(filepath=self.fbestweights,
                                             save_weights_only=True, save_best_only=True,
                                              monitor='val_loss', mode='min')]
        if patience_lr is not None:
            callbacks += [tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1)]

        if os.path.isfile(self.fbestweights):
            os.remove(self.fbestweights)
        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics = ['accuracy'])
        history = self.model.fit(X_train, y_train,
                                 epochs = epochs,
#                                steps_per_epoch = traingen.n//traingen.batch_size,
                                validation_data = (X_val, y_val),
#                                validation_steps = valgen.n//valgen.batch_size,
                                callbacks = callbacks, class_weight = class_weight)
        if os.path.isfile(self.fbestweights):
            self.model.load_weights(self.fbestweights)
        plot_history(f"{self.name}", history)
        self.prt("fit(): Fin\n")
        return history
        
    def predict(self, off_start, off_end, input_file=None):
        X_test = self.preprocess_X_test(off_start, off_end, input_file)
         
        self.prt("predict(): Début")
        softmaxout = self.model.predict(X_test, verbose = 1)
        y_pred = [self.fit_labels[i] for i in np.argmax(softmaxout, axis=1)]
        self.prt("predict(): Fin\n")
        return y_pred
    
    def save(self, nb=None):
        """ Sauvegarde sur disque """
        if not hasattr(self, 'fit_length') or self.fit_length == 0:
            self.prt("Il n'y a rien à sauvegarder, fit n'a pas été exécuté)")
            return None
        if nb is None:
            nb = self.fit_length
        filename = f"{self.name}_{nb}"
        fprefix = os.path.join(self.outdir, filename)
        if hasattr(self, "model") and hasattr(self.model, "save"):
            f = fprefix + "_model.hdf5"
            self.model.save(f)
            self.prt(f"Modèle sauvegardé dans {f}")
            self.model = None
        f = fprefix + "_object.pkl"
        tmpf = f + '_tmp'
        selfvars = vars(self)
        weaks = []
        for v in selfvars:
            try:
                pickle.dump(selfvars[v], open(tmpf, 'wb'))
            except:
                print("Pas de sauvegarde de ", v)
                weaks.append(v)
        for v in weaks:
            del(selfvars[v])
        if os.path.isfile(tmpf):
            os.remove(tmpf)
        pickle.dump(self, open(f, 'wb'))
        self.prt(f"Objet complet sauvegardé dans {f}")
        return f
    
    def evaluate(self, samples_number=-1,
                 test_size=TEST_SIZE,
                 val_split=VALIDATION_SPLIT,
                 off_train=0):
        """ Cycle complet fit + predict + report + save """
        if samples_number == -1: # On prend tous le  jeu d'entrainement
            samples_number = len(get_y()) - off_train
        self.nb = samples_number # référence des enregistrements fichier
        off_end = off_train + samples_number
        t0 = time.time()
        off_test = off_train + int(samples_number * (1 - test_size))
        off_val = off_train + int((off_test - off_train) * (1 - val_split))
        self.fit(off_train, off_val, off_test)
        y_test = get_y()[off_test : off_end]
        y_pred = self.predict(off_test, off_end)
        self.report(y_test, y_pred)
        t = int(time.time() - t0)
        self.prt(f"Evaluation exécutée en {t} secondes")
        self.save(self.nb)
    
    def create_final_model(self):
        """ Création du modèle """
        t0 = time.time()
        nb = len(get_y())
        self.fit(0, nb)
        self.save()
        df = pd.read_csv(X_TEST_CSV_FILE)
        y_pred = self.predict(0, df.shape[0], input_file=X_TEST_CSV_FILE)
        df['prdtypecode'] = y_pred
        csvfile = os.path.join(self.outdir, f"test_update_{self.name}.csv")
        df.to_csv(csvfile)
        t = int(time.time() - t0)
        self.prt(f"{csvfile} crée en {t} secondes")


class CatDataset(tf.keras.utils.Sequence):
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    Générateur de donnéees utilisé par un modèle à entrées multitples.
    Les données préprocessées Xs sont stockéees dans des fichiers qui
    seront lus à la demande (__getitem__). Chaque fichier contient les
    données complètes d'un batch.
    Par exemple, si on a trois sous modèles concaténés, chaque fichier
    de batch contiendra:
             <batch_size> échantillons preprocessés du mode X1 
             <batch_size> échantillons preprocessés du mode X2 
             <batch_size> échantillons preprocessés du mode X3 
    """
    def __batch_filepath(self, index):
        """ Fichier contenant les data du batch <index> """
        return os.path.join(self.dir, f"batch_{index}.npy")

    def __init__(self, name, batch_size, objs,
                 Xs, y=None, shuffle=False):
        """
        name:       nom du partitionnement
        batch_size: taille des batchs
        objs:       liste des instances associées à chaque sous-modèle
        Xs:         liste des données pré-processées de chaque sous modèle
        y:          liste globale des targets, ou None
        shuffle:    mélange des batchs à chaque époque
        """
        nb = len(Xs[0])
        self.y = y
        self.batch_size = batch_size
        self.input_number = len(Xs)
        self.batch_number = int(nb / batch_size)
        self.batch_indexes = range(self.batch_number)
        if shuffle:
            self.batch_indexes = sklearn.utils.shuffle(self.batch_indexes, random_state=1999)
        self.shuffle = shuffle

        self.dir = os.path.join(OUTDIR, f"{self.__class__.__name__}_{name}_{nb}")
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        else:
            print(f"Nettoyage de {self.dir}")
            for f in os.listdir(self.dir):
                os.remove(os.path.join(self.dir, f))

        print(f"Création de {self.batch_number} fichiers de batch dans {self.dir}")
        for i, X in enumerate(Xs):
            print(f"* modèle {objs[i].name}")
            isfile = type(X[0]) == str and os.path.isfile(X[0])
            for index in tqdm.tqdm(range(self.batch_number)):
                if not isfile:
                    X_batch = X[index*batch_size:(index+1)*batch_size, ...]
                else:
                    # Si ce sont des fichiers (images) on utilise la méthode
                    # du modèle pour les lire dans le bon format
                    X_batch = X[index*batch_size:(index+1)*batch_size]
                    imgs = list()
                    for f in X_batch:
                        img = objs[i].data_from_file(f)
                        imgs.append(img)
                    X_batch = np.array(imgs)
                batchfile = self.__batch_filepath(index)
                # Ouverture (création si besoin) du fichier de batch <index> et
                # écriture des données de type "X" à la suite des autres
                with open(batchfile, 'a+b') as f:
                    np.save(f, X_batch)

    def __len__(self):
        """ Retourne le nombre de batchs """
        return self.batch_number

    def __getitem__(self, index):
        """
        Retourne le batch <index>, de forme
        X = [(batchsize, X1..), (batchsize, X2..), ...] , y= (batch_size,)
        """
        index = self.batch_indexes[index]
        batchfile = self.__batch_filepath(index)
        X = list()
        # Ouverture du fichier de batch <index> et lecture des données
        # dans l'ordre où elles ont été écrites
        with open(batchfile, 'rb') as f:
            for _ in range(self.input_number):
                objbatch = np.load(f)
                X.append(objbatch)
        if self.y is None:
            return X
        else:
            y = self.y[index*self.batch_size : (index+1)*self.batch_size]
            return X, y

    def on_epoch_end(self):
        """ Changements effectués à chaque fin d'époque """
        if self.shuffle:
            self.batch_indexes = sklearn.utils.shuffle(self.batch_indexes, random_state=1968)


class RakutenCatModel(RakutenBaseModel):
    """
    Classe template pour les modèles qui concatenent plusieurs modèles de base.
    Elle utilise la classe CatDataset pour la génération des données
    
    Le modelè doit instancier les sous modèles utilisés dans la méthode fit()
    en passant le numéro de référence nb (nombre d'échantillons utilisés)
    Par exemple:
                self.objs = [ TextEmbed1(self.fit_length),
                              ImageXXXX(self.fit_length),
                              RNNTruc(self.fit_length)
                            ]
     Cela correspond à initialiser ces sous modèles avec leurs sauvegardes
     créés lors de leurs exécutions antérieures.
     il est ensuite possible de récupérer des informations sur chaque modèle:
     - ses méthodes de preprocessings à appliquer sur les données
     - sa structure de layers avec get_model_body()
     - les poids de ses layers entrainés

    """
    def __init__(self, name, nb=None):
        super().__init__(name, nb)

    def create_train_generators(self, off_start, off_val, off_end, input_file=None):
        """
        Création des générateurs de data pour l'entrainement et sa validation
        (a valeur de self.validation_split est utilisée pour les partitionner)
        """
        length = off_end - off_start
        self.prt(f"Preprocessing des {length} données d'entrainement")
        X_train = []
        for obj in self.objs:
            X = obj.preprocess_X_train(off_start, off_val, input_file)
#            X = sklearn.utils.shuffle(X, random_state=2021)
            X_train.append(X)
        y_train = self.preprocess_y_train(off_start, off_val, input_file)
        X_val = []
        for obj in self.objs:
            X = obj.preprocess_X_test(off_val, off_end, input_file)
#            X = sklearn.utils.shuffle(X, random_state=2021)
            X_val.append(X)
        y_val = self.preprocess_y_train(off_val, off_end, input_file)
#        y_train = sklearn.utils.shuffle(y_train, random_state=2021)
        
        self.prt(f"Instantiation des générateurs d'entrainement")
        traingen = CatDataset("train", self.batch_size, self.objs,
                              X_train, y_train, shuffle=True)
        valgen = CatDataset("val", self.batch_size, self.objs,
                            X_val, y_val, shuffle=True)
        return traingen, valgen

    def create_test_generator(self, off_start, off_end, input_file=None):
        """ Création du générateur des data de test """
        self.prt(f"Preprocessing des données de test")
        X_test = []
        for obj in self.objs:
            X_test.append(obj.preprocess_X_test(off_start, off_end, input_file))
        length = off_end - off_start

        self.prt(f"Instantiation du générateur de test")
        testgen = CatDataset("test", 1, self.objs, X_test)
        return testgen

    def copy_submodels_weights(self):
        """
        Copie des poids des layers des sous modèles de bases vers leurs clones
        (ayant le même nom) du modèle courant
        """
        print(f"  * Init des layers avec les poids des modèles de base:")
        for obj in self.objs:
            for oldlayer in obj.model.layers:
                weights = oldlayer.get_weights()
                if len(weights) > 0:
                    for newlayer in self.model.layers:
                        if newlayer.name == oldlayer.name:
                            print(f"      - {newlayer.name}")
                            newlayer.set_weights(weights)

    def predict(self, off_start, off_end, input_file=None):
        """
        Prédiction sur les données de input_file[off_start:off_end]
        """
        length = off_end - off_start
        self.prt(f"Prédiction pour {length} échantillons")
        testgen = self.create_test_generator(off_start, off_end, input_file)
         
        self.prt("predict(): Début")
        softmaxout = self.model.predict(testgen, verbose = 1)
        y_pred = [self.fit_labels[i] for i in np.argmax(softmaxout, axis=1)]
        self.prt("predict(): Fin\n")
        return y_pred
