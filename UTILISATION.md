# Guide d'utilisation

Ce guide explique ce que fait chaque notebook, ce que les résultats disent, et ce
qu'ils ne disent pas.

---

## Parcours

```mermaid
flowchart LR
    A[01 - Exploration<br/>et nettoyage] --> B[02 - Feature engineering<br/>geographique]
    B --> C[03 - Modelisation<br/>XGBoost + CV]
    C --> D[Modele + metriques]
```

---

## 1. Exploration et nettoyage — `01`

Analyse des distributions, des valeurs manquantes et des corrélations.

Deux décisions y sont prises, toutes deux commentées dans le notebook :

**`total_bedrooms` est imputé par la médiane** (≈ 1 % de valeurs manquantes). Une
imputation par médiane locale, à zone géographique équivalente, serait plus fine ;
le gain a été jugé marginal pour 1 % des lignes.

**Les lignes plafonnées à 500 001 $ sont retirées.** La cible du jeu d'origine est
écrêtée : toutes les valeurs supérieures ont été ramenées à ce plafond. Les
conserver apprendrait au modèle une valeur qui n'existe pas.

> **Conséquence à retenir :** le R² obtenu **n'est pas comparable** aux résultats
> publiés sur ce jeu, qui incluent généralement ces lignes. Retirer les extrêmes
> réduit la variance de la cible et modifie mécaniquement le R².

---

## 2. Feature engineering géographique — `02`

C'est ce qui distingue ce projet du tutoriel dont il part.

### Distances de Haversine

Distance orthodromique entre chaque district et les grandes villes californiennes.
Une distance euclidienne sur latitude/longitude serait fausse : à cette latitude, un
degré de longitude ne vaut pas un degré de latitude.

### Score gravitaire

```text
G = Σ ( population_ville / distance^α )
```

Une simple distance à Los Angeles ignore qu'un district peut être proche de
plusieurs villes moyennes. Le score gravitaire agrège l'influence de toutes les
villes : une métropole lointaine et une ville moyenne proche peuvent produire un
score comparable — ce qui reflète mieux la réalité économique.

### Les autres variables

| Famille | Exemples |
| --- | --- |
| Ratios structurels | pièces par foyer, ratio de chambres, population par foyer |
| Socio-économique | revenu par personne, revenu × âge du bâti |
| Zones | tranches nord/sud, est/ouest |
| Transformations | logarithmes des variables très asymétriques |
| Encodage | proximité de l'océan, ordinal |

Environ 37 variables sont produites, réduites ensuite à 15.

---

## 3. Modélisation — `03`

```python
cross_val_score(model, X_trainval, y_trainval, cv=5, scoring='r2')
RandomizedSearchCV(xgb_model, param_dist, n_iter=50, cv=5)
```

Validation croisée et recherche d'hyperparamètres sont ajustées **sur le train
uniquement**. Le jeu de test n'intervient qu'à l'évaluation finale.

### Résultats

| Métrique | Valeur |
| --- | ---: |
| R² validation croisée | 0,8317 |
| R² holdout | 0,8338 |
| RMSE holdout | ≈ 40 273 $ |
| Variables retenues | 15 |

**Le chiffre à regarder n'est pas le R², c'est l'écart.** 0,002 entre validation
croisée et holdout : le modèle généralise, il n'a pas surappris sa partition
d'entraînement. Un R² holdout nettement supérieur au R² CV serait plus suspect que
rassurant.

### RMSE : ce que valent 40 273 $

Sur des prix médians allant d'environ 15 000 à 500 000 $, une erreur type de 40 k$
signifie qu'une prédiction se situe typiquement à ±40 k$ du prix réel. Pour un
arbitrage d'investissement, c'est large ; pour une segmentation de marché, c'est
exploitable.

### La courbe R² / nombre de variables

![R² en fonction du nombre de variables](assets/r2_vs_features.png)

Le notebook retire les variables par importance croissante et mesure la
performance. La courbe s'aplatit vers 15–20 variables : au-delà, chaque variable
supplémentaire n'apporte plus rien de mesurable.

C'est ce qui justifie le modèle à 15 variables — plus compact, plus rapide, plus
interprétable, à performance équivalente.

---

## 4. Lecture géographique

![Carte des prix](assets/california_price_map.png)

Les prédicteurs les plus forts sont le **revenu médian**, puis la **localisation** :
distance aux grandes villes, proximité de l'océan, et les interactions
géographie × revenu.

Ce résultat n'a rien de surprenant — et c'est plutôt bon signe. Un modèle qui
placerait l'âge du bâti devant le revenu médian devrait éveiller la méfiance.

---

## 5. Ce que ce modèle ne permet pas

**Estimer un bien aujourd'hui.** Les données datent de **1990**. Prix, démographie
et géographie économique californiennes ont profondément changé.

**Estimer un logement individuel.** L'unité d'observation est le **district**, pas
la maison. Les valeurs sont des médianes de district.

**Conclure à une causalité.** Le modèle capte des associations. Il ne démontre pas
que rapprocher un district d'une ville augmenterait sa valeur.

**Se comparer aux benchmarks publics.** Les lignes plafonnées ayant été retirées, le
R² n'est comparable qu'à lui-même.

---

## 6. Une limite de méthode à connaître

La sélection des 15 variables ajuste un modèle sur le jeu **complet**, test compris.
Le classement d'importance intègre donc de l'information du test, et **le R²
annoncé est optimiste**.

La correction consiste à placer la sélection à l'intérieur de la boucle de
validation croisée. Détail dans
[ARCHITECTURE.md](ARCHITECTURE.md#5-le-biais-de-sélection-et-pourquoi-il-compte).

Une seconde limite, propre aux données géographiques : le découpage
entraînement/test est **aléatoire, pas spatial**. Deux districts voisins se
retrouvent de part et d'autre du découpage et se ressemblent, ce qui gonfle la
performance apparente. Une validation par blocs géographiques serait plus
exigeante.

---

## 7. La démonstration navigateur

<https://juleescourne.github.io/portfolio-data-analyst/#/housing>

La carte interactive affiche un **score relatif 0–100** comparant les zones dans un
scénario donné, coloré par **rang centile** — le score brut est trop asymétrique
pour une échelle linéaire, qui écraserait la quasi-totalité des points.

Ce n'est **pas une estimation monétaire calibrée**. Les métriques réelles du modèle
sont celles de ce dépôt.
