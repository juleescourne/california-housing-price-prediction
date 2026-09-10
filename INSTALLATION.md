# Installation

Temps nécessaire : **5 minutes**, hors téléchargement du jeu de données.

## Prérequis

| Outil | Version | Vérifier |
| --- | --- | --- |
| Python | 3.11 ou supérieur | `python --version` |
| Compte Kaggle | gratuit | pour télécharger le jeu de données |

---

## 1. Récupérer le projet

```bash
git clone https://github.com/juleescourne/california-housing-price-prediction.git
cd california-housing-price-prediction
```

## 2. Environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate          # .\.venv\Scripts\Activate.ps1 sous Windows
pip install -r requirements.txt
```

> **GeoPandas peut poser problème sous Windows.** Il dépend de bibliothèques
> géospatiales compilées (GDAL, GEOS, PROJ). En cas d'échec de `pip`, passez par
> conda :
>
> ```bash
> conda install -c conda-forge geopandas contextily
> ```
>
> GeoPandas et Contextily ne servent qu'aux fonds de carte. Le reste de la chaîne
> — nettoyage, feature engineering, modélisation — fonctionne sans eux.

## 3. Télécharger le jeu de données

<https://www.kaggle.com/datasets/camnugent/california-housing-prices>

Placez le fichier ici :

```text
data/raw/housing.csv
```

20 640 districts californiens, recensement de 1990.

## 4. Exécuter les notebooks

```bash
jupyter notebook
```

Dans l'ordre — chacun produit l'entrée du suivant :

```text
notebooks/01_data_exploration_cleaning.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_modeling_xgboost.ipynb
```

> Le notebook 03 exécute une `RandomizedSearchCV` de 50 tirages × 5 plis. Comptez
> **plusieurs minutes** selon votre machine ; c'est normal.

---

## Ce que produit une exécution complète

```text
data/preprocessed/01_housing_clean.csv    jeu nettoyé
data/preprocessed/02_housing_model.csv    jeu enrichi, prêt à modéliser
models/xgb_housing_model.pkl              modèle entraîné
models/feature_names.json                 variables retenues
models/model_info.json                    métriques et hyperparamètres
```

Ces répertoires sont exclus de Git.

---

## Vérifier l'installation

Après le notebook 03, contrôlez la cohérence des métriques enregistrées :

```bash
python -c "import json; print(json.load(open('models/model_info.json')))"
```

Vous devez y voir un `r2_test` **calculé**, proche de 0,83, et non une valeur
figée : la fonction d'entraînement renvoie ses métriques, précisément pour qu'elles
ne puissent pas se désynchroniser du modèle enregistré.

---

## Problèmes courants

**`FileNotFoundError: data/raw/housing.csv`**
Le fichier Kaggle n'est pas au bon endroit, ou porte un autre nom.

**Échec d'installation de GeoPandas**
Voir l'encadré de l'étape 2. Les cartes sont optionnelles.

**`NameError: name 'X_test' is not defined` dans le notebook 03**
Les jeux de test sont locaux aux fonctions d'entraînement. Les métriques transitent
par la valeur de retour, pas par des variables globales. Exécutez les cellules dans
l'ordre.

**La recherche d'hyperparamètres est très lente**
Réduisez `n_iter` dans `RandomizedSearchCV`, ou fixez `n_jobs=-1` si ce n'est pas
déjà le cas.

**Les notebooks s'ouvrent lentement sur GitHub**
Les notebooks 01 et 02 embarquent leurs figures : 1,4 Mo et 808 Ko. C'est
volontaire — un relecteur voit les résultats sans rien exécuter.
