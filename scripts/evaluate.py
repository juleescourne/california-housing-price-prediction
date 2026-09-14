"""Train-only preprocessing and geographic holdout on raw California Housing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

def evaluate(path, output):
    df = pd.read_csv(path)
    required = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                'population', 'households', 'median_income', 'ocean_proximity', 'median_house_value']
    if not set(required) <= set(df): raise ValueError('Missing California Housing columns')
    df = df[required].copy()
    for column in [c for c in required if c != 'ocean_proximity']:
        df[column] = pd.to_numeric(df[column], errors='raise')
        if np.isinf(df[column]).any(): raise ValueError(f'Infinite values in {column}')
    if df[['longitude', 'latitude', 'median_house_value']].isna().any().any():
        raise ValueError('Coordinates and target must be complete')
    if not df.longitude.between(-180, 180).all() or not df.latitude.between(-90, 90).all():
        raise ValueError('Coordinates outside geographic bounds')
    X, y = df.drop(columns='median_house_value'), df.median_house_value
    groups = (np.floor(df.latitude).astype(str)+'_'+np.floor(df.longitude).astype(str)).to_numpy()
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X.iloc[trainval], y.iloc[trainval], groups[trainval]))
    train, validation = trainval[a], trainval[b]
    partitions = [set(groups[index]) for index in [train, validation, test]]
    if any(partitions[a] & partitions[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('Geographic groups overlap between partitions')
    numeric = X.select_dtypes(include='number').columns.tolist()
    def preprocessing():
        return ColumnTransformer([('numeric', SimpleImputer(strategy='median'), numeric),
                                  ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['ocean_proximity'])])
    fitted, validation_rmse = {}, {}
    for depth in [3, 5]:
        model = make_pipeline(preprocessing(), XGBRegressor(n_estimators=250, max_depth=depth,
                              learning_rate=.05, subsample=.9, colsample_bytree=.9, n_jobs=2, random_state=42))
        model.fit(X.iloc[train], y.iloc[train])
        fitted[str(depth)] = model
        validation_rmse[str(depth)] = float(np.sqrt(mean_squared_error(y.iloc[validation], model.predict(X.iloc[validation]))))
    best = min(validation_rmse, key=validation_rmse.get)
    models = {'dummy_mean': DummyRegressor().fit(X.iloc[train], y.iloc[train]), 'xgboost': fitted[best]}
    scores = {}
    for name, model in models.items():
        predicted = model.predict(X.iloc[test])
        scores[name] = dict(rmse=float(np.sqrt(mean_squared_error(y.iloc[test], predicted))), r2=float(r2_score(y.iloc[test], predicted)))
    report = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(df),
                  capped_rows=int((y >= 500001).sum()), features=X.columns.tolist(),
                  split=dict(train=len(train), validation=len(validation), test=len(test), seed=42,
                             grouping='1 degree latitude/longitude blocks', train_groups=sorted(set(groups[train])),
                             validation_groups=sorted(set(groups[validation])), test_groups=sorted(set(groups[test]))),
                  validation_rmse=validation_rmse, selected_max_depth=int(best), test=scores,
                  versions=dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__),
                  limitations=['1990 data, unsuitable for current property valuation.',
                               'Blocks share borders; residual spatial dependence remains.',
                               'Capped target retained; predicting recorded medians.',
                               'Historical notebooks and browser model are not this evaluation.',
                               'This holdout is not an external dataset.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/raw/housing.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/evaluation.json'))
    args = parser.parse_args()
    evaluate(args.data, args.output)
