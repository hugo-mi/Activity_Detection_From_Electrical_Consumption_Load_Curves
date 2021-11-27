# Projet Fil Rouge : Détection d'activité : Analyse d'une courbe de charge

**_Projet fil rouge dans le cadre de la formation à Télécom Paris en lien avec l'entreprise Baalbek Management_**

## Contexte

Le sujet de la détection d'activité est un sujet porté par de nombreux acteurs économiques et qui peut être source de nombreux services aux usagers.

Notamment :
* Sécurisation des logements notamment les logements secondaires ou les logements mis en location temporaire
* Prolonger l'autonomie des seniors en leur permettant un maintient à domicile, en alertant les proches ou le personnel médical en cas d’absence anormal d’activité
* Détection de l’heure du retour de l’école des enfants

Les solutions actuelles sont des solutions intrusives, pas forcément acceptées ni simples à installer: par exemple via l'installation de différent type de capteurs qui vont monitorer la présence des habitants etc...

## Objectif

Travailler sur les technologies **non-intrusives** de détection d'activité à partir des courbes de charge de consommation électrique

## Formulation de la problématique

Typiquement, on peut distinguer trois catégories de consommation:

* Consommation de tous les appareils en veille

* Consommation des équipements qui se déclenchent et s'éteignent seuls (type frigo, ballon d'eau chaude... etc

* Consommation des équipements déclenchés par l'utilisateur

Seule cette dernière catégorie d'équipement permet de détecter une réelle activité dans le logement.

Il s'agit donc de concevoir des algorithmes de Machine Learning qui vont permettre de faire une **classification binaire** prédisant l'activité du logement, avec la contrainte de pouvoir s'adapter à différent types de logements pour lesquels on n'a pas de données labelisées (unsupervised learning)

## Data

* Dataset Open Source de courbe de charge labellisées

## Métriques d'évaluation

* Accuracy
* Recall
* F-Score
