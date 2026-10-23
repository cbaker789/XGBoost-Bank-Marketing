"""Artifact integration checks; skipped until the research run has generated a model."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from research import ROOT, load_data, infer, budget_metrics

READY = (ROOT/'outputs/research/selected_bundle.joblib').exists()

@unittest.skipUnless(READY, 'Run run_experiments.py to enable artifact integration checks.')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT/'outputs/research/selected_bundle.joblib')
        cls.frame = load_data()
        cls.pred = pd.read_csv(ROOT/'outputs/research/final_predictions.csv')
        cls.metrics = json.loads((ROOT/'results/final_evaluation.json').read_text())['selected_model']

    def test_reloaded_inference_reproduces_saved_probabilities(self):
        rows = self.pred.source_row.to_numpy()
        actual = infer(self.bundle,self.frame.iloc[rows].drop(columns='y'))
        np.testing.assert_allclose(actual,self.pred.probability,atol=1e-7)
        np.testing.assert_array_equal(self.pred.actual,(self.frame.iloc[rows].y=='yes').astype(int))

    def test_published_final_metrics_match_predictions(self):
        self.assertAlmostEqual(average_precision_score(self.pred.actual,self.pred.probability),self.metrics['average_precision'])
        self.assertAlmostEqual(roc_auc_score(self.pred.actual,self.pred.probability),self.metrics['roc_auc'])
        self.assertAlmostEqual(log_loss(self.pred.actual,self.pred.probability),self.metrics['log_loss'])

    def test_decomposition_reconciles_to_observed_rate_difference(self):
        values = pd.read_csv(ROOT/'results/rate_decomposition.csv')
        summed = values[['mix_component','within_group_component','unmatched_category_component']].sum(axis=1)
        np.testing.assert_allclose(summed,values.rate_change,atol=1e-12)

    def test_contact_budget_table_reconciles(self):
        for row in pd.read_csv(ROOT/'results/contact_budget_scenarios.csv').itertuples():
            recalculated = budget_metrics(self.pred.actual,self.pred.probability,row.budget)
            self.assertEqual(recalculated['contacts'],row.contacts)
            self.assertAlmostEqual(recalculated['expected_subscribers'],row.expected_subscribers)

    def test_transformer_learned_training_numeric_means_only(self):
        if self.bundle['name'] != 'logistic':
            self.skipTest('Numeric scaling mean check applies to selected logistic regression.')
        a,b = self.bundle['partitions']['train']
        columns = self.bundle['preprocessor'].transformers_[1][2]
        expected = self.frame.iloc[a:b][columns].mean().to_numpy()
        np.testing.assert_allclose(self.bundle['preprocessor'].named_transformers_['numeric'].mean_,expected)

@unittest.skipUnless(READY, 'Generate model artifacts first.')
class DemoTests(unittest.TestCase):
    def test_demo_loads_scores_samples_and_updates_budget(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        app.slider[0].set_value(20).run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any(m.value=='1,236' for m in app.metric))

if __name__ == '__main__':
    unittest.main()
