# Approche non supervisée : Auto-Encodeur Convolutionnel (AEC) pour la détection d'anomalies

Avec l’approche Auto-encodeur, l’objectif est de modéliser le comportement normale des activités quotidiennes d’un foyer. Pour ce faire, l’auto-encodeur doit pouvoir apprendre la courbe de charge de consommation électrique normale.

Pour détecter une activité anormale, la consommation électrique de base doit être identifiée.

Pour obtenir cette courbe de charge de consommation normale synonyme d’inactivité dans un foyer, deux possibilités s’offrent à nous :

- Se référer à la courbe de charge nocturne (heures creuses)

- Sélectionner individuellement la consommation électrique de chacun des appareils permettant de prédire une activité 
    - => Approche moins généralisable car on ne connaît jamais à l’avance le nombre et le type d’appareils électriques utilisés au sein d’une maison. Mais cette méthode va mieux coller à nos données donc on pense avoir de meilleure résultat.

# Méthodologie

## Pre-processing

Dans notre cas nous l’étape de preprocessing est une étape cruciale. Donc on a décidé de rendre la phase de preprocessing entièrement paramétrable. Plusieurs paramètres qui régissent la création des séquences d’entraînement et de test doivent être choisis en amont. Cela permet alors au modèle de gagner en flexibilité mais aussi de pouvoir s’adapter au comportement de la consommation électrique quotidienne de n’importe quel foyer.

Les paramètres sont les suivants :

- ``TIME_STEP`` qui définit la durée d’échantillonnage de la courbe électrique
- ``DURATION_TIME`` : qui fixe la durée d’une séquence
- ``OVERLAP_PERIOD_PERCENT`` : qui correspond taux de chevauchement des données entre chaque séquence
- ``TIMEFRAMES`` : qui définit la plage horaire des heure creuses (ce paramètre est crucial pour extraire la courbe de consommation électrique de base)
- ``SPLIT_METHOD`` = méthode pour construire la séparation du jeu d’entraînement et de test (ex choisir des jour aléatoirement) 

Tout l’intérêt du preprocessing est de s’adapter au mieux à la routine quotidienne d’une famille pour pouvoir bien extraire la courbe de charge de base (courbe de charge nocturne)

**Train-Test split méthode**
Pour construire notre jeu de test, on tire à aléatoirement **20%** des jours qui composent notre dataset. Les jours restant forment le jeu d'entraînement. 

Ensuite le preprocessing construit les différentes séquences qui se chevauchent. Par exemple, la sortie du preprocessing est un tableau en 3D [samples, sequence_length, features]

## Architecture de l'AEC

L’architecture de l’auto-encodeur se compose d’une succession de couches de convolution puis déconvolution pour obtenir la reconstruction de l’entrée.

## Entraînement de l'AEC

**Objectif :**
Nous avons utilisé le jeu de données d’entraînement ``X_train`` à la fois comme entrée et comme variable cible puisqu’il s’agit d’un mode de reconstruction.

**Pourquoi MSE ?**
L’objectif est de minimiser la fonction de coût de reconstruction. On a fait le choix d’utiliser cette fonction de perte car le but est de pénaliser les valeurs aberrantes étant donné que nous souhaitons apprendre la courbe de consommation de base.

Nous utilisons également un early stopping pour diminuer le temps d'entraînement lorsque la fonction de perte ne diminue pas.

## Visualisation de la courbe de reconstruction

On peut observer comment l’AE apprend à reconstruire la première séquence de notre jeu d'entraînement.

On remarque que le à reconstruire la courbe de charge de base qui lui a été donnée en entrée.

## Détection des anomalies (i.e activité)

Pour détecter les anomalies, 

Premièrement, on calcul la perte **MAE** sur les échantillons d’entraînement.

Ensuite, on identifie la valeur maximale de la perte **MAE** . Cela correspond à la plus mauvaise performance de notre modèle en essayant de reconstruire un échantillon. Cela constitue le seuil (i.e threshold ) pour la détection des anomalies.

Enfin, si la perte de reconstruction d’un échantillon est supérieure à cette valeur seuil, alors nous pouvons en déduire que le modèle fait face à un comportement qui ne lui est pas familier. Nous allons étiqueter cet échantillon (i.e séquence) comme une anomalie

## Visualisation du threshold sur ``X_train`` et ``X_test``


A ce stade, le modèle à appris à reconstruire la courbe de charge de base. Cela signifie que le modèle a appris à modéliser le comportement normal des activités quotidiennes d’un foyer. Sur cette base, la détection des anomalies peut être réalisée en évaluant tout simplement les écarts entre la courbe de charge de base apprise par le modèle et la courbe de charge quotidienne d’un foyer qui comprend des pics d’activité (i.e surconsommation électrique). Cet écart est calculé avec la fonction de perte Mean Absolute Error. Nous avons choisi cette fonction de perte car elle est plus robuste aux données aberrantes ce qui correspond dans notre cas aux anomalies. Sur les différents histogrammes, on voit l’évolution de cet écart pour le jeu d'entraînement et le jeu de test.

## Post-processing

Le but du pré-processing est d’affiner les prédictions du modèle car il arrive parfois qu’un point de
donnée se trouve à la fois dans une séquence prédite comme étant une anomalie et une séquence prédite
n’étant pas une activité

IMAGE

Par exemple, le point de données encadré en rouge sur l’image ci-dessus correspond à une donnée
de consommation à la date du 26 avril 2016 à 06 :36 :00. Ce point de données appartient à plusieurs
séquences qui se sont chevauchées à savoir la séquence n° 109 qui est prédite en non activité et les
séquences n° 105, 106, 107 et 108 qui sont prédites comme étant des activités. Ainsi par vote majoritaire,
le postprocessing prédit une activité pour cette date. Cela doit correspondre à la l’heure où la personne
se lève le matin avant de se rendre au travail et prend une douche ou bien encore son petit déjeune

## Evaluation du modèle

**Matrice de confusion**

**Activité prédite VS Activité réelle**

**IoU Threshold**

## Discussion de l'approche




## Résultat

