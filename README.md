# Rakuten_py

## Contexte

Ce projet s’inscrit dans le challenge Rakuten France Multimodal Product Data Classification: Il s’agit de prédire le code type de produits (tel que défini dans le catalogue Rakuten France) à partir d’une description texte et d’une image.

Rakuten France souhaite pouvoir catégoriser ses produits automatiquement grâce à la désignation, la description et les images des produits vendus sur son site.

## Déroulement du projet

Le projet suivait un plan en plusieurs étapes :
* Chargement et première exploration des données, visualisation et analyses statistiques.
* Modélisation de différents algoritmes de classification de texte comme, linear SVC, logistic régression, KNN.
* Modélisation d’algorithmes de  réduction de dimensions PCA pour la classification d'images.
* Modélisation d'algorithmes Deep Learning:
  * Réseau de neurones convolutifs (ConvNet) pour la classification d'images,
  * Réseaux de neurones récurrents (RNN) pour la classification de texte.
  * Concatenation des meilleurs modèles textuels et des modèles images

## Modèle choisi

Le modèle final est issu de la sélection puis concaténation des meilleurs modèles individuels texte et image. La sélection des modèles individuels est elle même issue d’un processus itératif multi boucles de nos différents modèles machine learning et Deep Learning texte et images.
Le détail de cette étude et des codes réalisés par l’équipe au cours du projet se trouve dans ce GitHub.
Les résultats de scoring f1 weighted que nous avons obtenus sont de 0.8783.

![Captura de Pantalla 2021-11-13 a les 16 09 24 3188](https://user-images.githubusercontent.com/8598179/141648932-5ca4f7f9-8ddd-46d6-b96b-f33dc815af3b.png)

Par ailleurs, nous avons **soumis nos prédictions** le 9 août 2021 sur le site du challenge, et notre équipe FEEEScientest a atteint un score de 0.8628, ce qui nous classe actuellement dans les **10 premières équipes**.
