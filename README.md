# California Housing : évaluation géographique d'une régression

Prédire la valeur médiane enregistrée d'un district à partir du recensement de
1990. Une évaluation par blocs géographiques compare XGBoost à une constante.

## Résultats reproductibles

| Modèle | RMSE test ($) | R² test |
| --- | ---: | ---: |
| Moyenne du train | 122083 | -0.0274 |
| XGBoost | 69277 | 0.6692 |

20 640 observations conservées, dont 965 valeurs plafonnées. Blocs de 1° séparés
entre train (10 540 lignes), validation (3 870) et test (6 230). Les proportions
portent sur les blocs et donnent des effectifs inégaux. Deux profondeurs (3, 5)
sont comparées sur validation ; la profondeur 5 est retenue avant lecture du test.

[Rapport et empreinte des données](reports/evaluation.json)
· [Script exécuté](scripts/evaluate.py)

## Reproduire

Python **3.12 ou supérieur** est nécessaire pour les versions figées dans `requirements-eval.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
python scripts/evaluate.py --data data/raw/housing.csv
```

Source utilisée : [`housing.csv` du dépôt pédagogique d'Aurélien Géron](https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv),
également disponible via le [dataset Kaggle historique](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
CSV non redistribué ; empreinte enregistrée dans le rapport.

## Vérification automatisée

```bash
python -m unittest discover -s tests -v
```

La CI exécute le script sur un petit jeu synthétique : elle vérifie les contrats
d’entrée et la cohérence du protocole, sans téléchargement Kaggle. Ces tests ne
recalculent pas les scores du rapport sur les données publiques.

## Corrections méthodologiques

- Neuf variables brutes fixées à l'avance, sans sélection sur le jeu complet.
- Imputation et encodage ajustés sur train uniquement.
- Blocs du test absents du train et de la validation.
- Observations plafonnées conservées ; baseline explicite.

## Exploration géographique et démonstration

Les notebooks historiques développent Haversine, score gravitaire et interactions,
mais sélectionnaient quinze variables sur le jeu complet. R² 0,8338 et RMSE
40 273 $ sont donc **exploratoires**. Les populations, variables et protocoles
diffèrent : ne pas comparer directement ces valeurs au nouveau rapport.
L'évaluation sur variables brutes ne démontre pas un gain du feature engineering.

[La démo navigateur](https://juleescourne.github.io/portfolio-data-analyst/#/housing)
conserve un modèle historique distinct. Son score relatif est normalisé à nouveau
pour chaque scénario : un score identique entre scénarios n'implique pas une
valeur brute identique. Il ne s'agit pas d'un prix calibré.

## Limites

Données de 1990 inadaptées à une estimation actuelle. Le modèle prédit la valeur
médiane enregistrée et plafonnée, pas la valeur économique non censurée. Les blocs
peuvent se toucher : une dépendance spatiale résiduelle demeure. Ce découpage
reproductible ne remplace pas une validation externe.

[Installation](INSTALLATION.md) · [Utilisation](UTILISATION.md) · [Architecture historique](ARCHITECTURE.md)

Code sous [licence MIT](LICENSE). Jules Courné.
