"""Geographic isolation and input-contract tests using synthetic districts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'data.csv'
        self.output = Path(self.tmp.name) / 'report.json'
        rng = np.random.default_rng(27)
        n = 240
        self.data = pd.DataFrame({
            'longitude': -124.2 + np.repeat(np.arange(20) % 5, 12),
            'latitude': 33.2 + np.repeat(np.arange(20) // 5, 12),
            'housing_median_age': rng.integers(1, 50, n),
            'total_rooms': rng.integers(100, 1000, n),
            'total_bedrooms': rng.integers(50, 200, n).astype(float),
            'population': rng.integers(100, 2000, n), 'households': rng.integers(50, 200, n),
            'median_income': rng.uniform(1, 10, n),
            'ocean_proximity': rng.choice(['INLAND', 'NEAR OCEAN'], n),
            'median_house_value': rng.uniform(50000, 450000, n)})
        self.data.loc[0, 'total_bedrooms'] = np.nan

    def run_evaluation(self):
        self.data.to_csv(self.source, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(self.source, self.output)
        return json.loads(self.output.read_text())

    def test_all_partitions_have_disjoint_complete_geographic_groups(self):
        report = self.run_evaluation()
        split = report['split']
        groups = [set(split[k + '_groups']) for k in ['train', 'validation', 'test']]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(groups[a] & groups[b])
        expected = set(np.floor(self.data.latitude).astype(str) + '_' + np.floor(self.data.longitude).astype(str))
        self.assertEqual(set.union(*groups), expected)
        self.assertEqual(sum(split[k] for k in ['train', 'validation', 'test']), len(self.data))
        self.assertEqual(set(report['test']), {'dummy_mean', 'xgboost'})
        self.assertTrue(all(np.isfinite(score['rmse']) for score in report['test'].values()))

    def test_missing_coordinate_is_rejected_instead_of_forming_nan_group(self):
        self.data.loc[0, 'latitude'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Coordinates and target'):
            self.run_evaluation()
        self.assertFalse(self.output.exists())

    def test_infinite_target_is_rejected(self):
        self.data.loc[0, 'median_house_value'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            self.run_evaluation()

    def test_non_numeric_feature_is_not_silently_dropped(self):
        self.data['median_income'] = self.data.median_income.astype(str)
        self.data.loc[0, 'median_income'] = 'unknown'
        with self.assertRaises(ValueError):
            self.run_evaluation()
