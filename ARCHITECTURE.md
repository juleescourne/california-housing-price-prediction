# Architecture et spécifications techniques

Ce document décrit la chaîne de traitement, l'enrichissement géographique — le cœur
du projet — la stratégie de validation et ses limites.

---

## 1. Chaîne de traitement

```mermaid
flowchart LR
    A[housing.csv<br/>20 640 districts] --> B[01 - Exploration<br/>et nettoyage]
    B --> C[02 - Feature engineering<br/>geographique]
    C --> D[03 - Modelisation<br/>XGBoost + CV]
    D --> E[Modele + metriques]
```

Chaque notebook écrit le fichier consommé par le suivant.

---

## 2. Nettoyage et parti pris sur le plafond

Le jeu de données de 1990 comporte deux particularités traitées explicitement.

### Valeurs manquantes

`total_bedrooms` est manquant sur environ 1 % des lignes, imputé par la médiane.

> **Limite.** La médiane est calculée sur le jeu **complet**, avant découpage. C'est
> une fuite statistique mineure mais réelle : une imputation correcte s'ajuste sur
> le train seul, dans un `ColumnTransformer`.

### Le plafond artificiel

La cible `median_house_value` est **écrêtée à 500 001 $** : toutes les valeurs
supérieures ont été ramenées à ce plafond lors de la constitution du jeu d'origine.

Ces lignes sont **retirées**. Conservées, elles apprendraient au modèle une valeur
qui n'existe pas dans la réalité, et les résidus dans le haut de la distribution
n'auraient aucun sens.

> **Conséquence à connaître.** Le R² obtenu **n'est pas comparable** aux résultats
> publiés sur ce jeu, qui incluent généralement les lignes plafonnées. Retirer les
> valeurs extrêmes réduit la variance de la cible et modifie mécaniquement le R².
> Le chiffre n'est comparable qu'à lui-même.

---

## 3. Enrichissement géographique

C'est la partie qui distingue ce projet du tutoriel dont il part.

### Distances aux grandes villes

Distance de Haversine — distance orthodromique sur une sphère — entre chaque
district et les grandes villes californiennes.

$$d = 2R \arcsin\sqrt{\sin^2\frac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\frac{\Delta\lambda}{2}}$$

Une distance euclidienne sur latitude/longitude serait fausse : à la latitude de la
Californie, un degré de longitude vaut environ 0,79 degré de latitude en kilomètres.

### Score d'attractivité gravitaire

Une simple distance à Los Angeles ignore qu'un district peut être proche de
plusieurs villes moyennes. Le score gravitaire, emprunté aux modèles d'interaction
spatiale, agrège l'influence de toutes les villes :

$$G = \sum_{i} \frac{P_i}{d_i^{\,\alpha}}$$

où $P_i$ est la population de la ville $i$ et $d_i$ la distance. Une grande ville
lointaine et une ville moyenne proche peuvent produire un score comparable — ce qui
correspond mieux à la réalité économique qu'une distance unique.

### Autres variables

| Famille | Exemples |
| --- | --- |
| Distances | distance à chaque ville, distance à la plus proche |
| Zones | tranches nord/sud et est/ouest |
| Ratios structurels | pièces par foyer, ratio de chambres, population par foyer |
| Socio-économique | revenu par personne, interactions revenu × âge du bâti |
| Interactions | géographie × revenu |
| Transformations | logarithmes des variables très asymétriques |
| Encodage | proximité de l'océan, encodage ordinal |

Le découpage en tranches utilise `pd.qcut`, dont les bornes sont calculées sur le
jeu complet — même limite que l'imputation.

---

## 4. Modélisation et validation

| Étape | Choix |
| --- | --- |
| Algorithme | `XGBRegressor` |
| Découpage | 80 % entraînement+validation / 20 % test |
| Validation | `cross_val_score`, 5 plis, sur le train uniquement |
| Hyperparamètres | `RandomizedSearchCV`, 50 tirages × 5 plis, sur le train uniquement |
| Sélection de variables | réduction de ~37 à 15 variables |

C'est le seul projet ML de mon portfolio avec une **vraie recherche
d'hyperparamètres versionnée** : la grille et la procédure figurent dans le
notebook, pas seulement leur résultat.

### Résultats

| Métrique | Valeur |
| --- | ---: |
| R² validation croisée | 0,8317 |
| R² holdout | 0,8338 |
| RMSE holdout | ≈ 40 273 $ |

L'écart entre R² CV et R² holdout est de 0,002 : le modèle généralise, il ne
surapprend pas la partition d'entraînement.

### La courbe R² en fonction du nombre de variables

Le notebook mesure la performance en retirant les variables par importance
croissante. La courbe s'aplatit vers 15–20 variables : au-delà, chaque variable
supplémentaire n'apporte plus rien de mesurable.

C'est ce qui justifie la réduction à 15 variables — un modèle plus compact,
plus rapide et plus interprétable, à performance équivalente.

---

## 5. Le biais de sélection, et pourquoi il compte

La cellule de sélection des 15 variables ajuste un modèle sur le **jeu complet** :

```python
model_temp.fit(X, y)          # X et y contiennent le jeu de test
importance_df = ...
top_15_features = importance_df['feature'].head(15).tolist()
```

Le classement d'importance intègre donc l'information du jeu de test. Le modèle
final est ensuite entraîné et évalué sur un découpage de ces 15 variables :
**le R² annoncé est optimiste**.

### Ce qu'il faudrait faire

Placer la sélection **à l'intérieur** de la boucle de validation croisée :
sélectionner sur chaque pli d'entraînement, évaluer sur le pli de validation. Puis
une seule évaluation finale sur un test resté intact — ou une validation croisée
imbriquée.

L'ampleur du biais est probablement modérée ici : XGBoost est peu sensible aux
variables faiblement informatives, et la courbe R²/variables est plate dans cette
zone. Mais « probablement modérée » n'est pas une mesure.

---

## 6. Métriques cohérentes avec le modèle

Le fichier `models/model_info.json` portait un R² **écrit en dur** :

```python
'r2_test': 0.8338,          # valeur littérale
```

alors que `r2_cv` était calculé. Les métriques enregistrées pouvaient donc se
désynchroniser silencieusement du modèle sauvegardé.

La fonction d'entraînement renvoie désormais ses métriques, consommées à
l'enregistrement :

```python
metrics = {'r2_test': float(r2_test), 'rmse_test': float(rmse_test)}
return X, best_model, random_search, metrics
```

---

## 7. Limites connues

- **Le jeu date de 1990.** Le modèle ne vaut rien pour une estimation actuelle : les
  prix, la démographie et la géographie économique californienne ont changé.
- **Sélection de variables biaisée** (section 5) : le R² est optimiste.
- **Imputation et découpage en tranches ajustés sur le jeu complet.**
- **R² non comparable aux benchmarks publics**, les lignes plafonnées ayant été
  retirées.
- **Découpage aléatoire, pas spatial.** Sur des données géographiques, deux
  districts voisins se retrouvent de part et d'autre du découpage et se
  ressemblent : c'est une forme d'autocorrélation spatiale qui gonfle la
  performance apparente. Une validation par blocs géographiques serait plus
  exigeante et plus honnête.
- **Variables absentes** : qualité des écoles, criminalité, accès aux transports,
  taille des parcelles, dynamique temporelle.
- **Aucune analyse des résidus par zone géographique**, alors que le projet est
  centré sur la géographie.
