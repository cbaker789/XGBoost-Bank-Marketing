import unittest
import numpy as np
import pandas as pd
from research import features, temporal_parts, budget_metrics, tune_threshold, apply_calibration

class ResearchTests(unittest.TestCase):
    def test_outcome_and_call_duration_are_excluded(self):
        x = features(pd.DataFrame({'duration':[100], 'y':['yes'], 'age':[40]}),'xgb_base')
        self.assertEqual(list(x.columns),['age'])

    def test_no_economic_features_survive_ablation(self):
        x = pd.DataFrame({c:[1.] for c in ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']})
        x['age'] = 40
        self.assertEqual(list(features(x,'xgb_no_economics')),['age'])

    def test_no_overlap_or_future_rows_in_temporal_partitions(self):
        parts = temporal_parts(1000,[.45,.55,.65])
        ranges = [list(range(s.start,s.stop)) for s in parts.values()]
        joined = sum(ranges,[])
        self.assertEqual(joined,list(range(650)))
        self.assertEqual(len(joined),len(set(joined)))
        self.assertEqual(parts['evaluation'].start,550)

    def test_invalid_windows_fail(self):
        with self.assertRaises(ValueError):
            temporal_parts(100,[.5,.5,.8])

    def test_budget_scores_handle_ties_without_label_order_bias(self):
        result = budget_metrics([1,1,0,0],[.5,.5,.5,.5],.25)
        self.assertEqual(result['contacts'],1)
        self.assertEqual(result['expected_subscribers'],.5)
        self.assertEqual(result['lift'],1)
        permuted = budget_metrics([0,1,0,1],[.5,.5,.5,.5],.25)
        self.assertEqual(result,permuted)

    def test_perfect_ranking_and_full_budget_reconcile(self):
        perfect = budget_metrics([0,1,0,1],[.1,.9,.2,.8],.5)
        self.assertEqual(perfect['capture'],1)
        self.assertEqual(perfect['lift'],2)
        full = budget_metrics([0,1,0,1],[.1,.9,.2,.8],1)
        self.assertEqual(full['expected_subscribers'],2)
        self.assertEqual(full['lift'],1)

    def test_policy_threshold_can_recover_low_probability_positives(self):
        y, p = np.array([0,0,1,1]),np.array([.01,.02,.15,.2])
        t = tune_threshold(y,p)
        self.assertGreater(t,.02)
        self.assertLessEqual(t,.15)
        np.testing.assert_array_equal(p>=t,y.astype(bool))

    def test_raw_calibration_identity(self):
        p = np.array([.1,.8])
        np.testing.assert_array_equal(apply_calibration(None,p),p)

    def test_never_contacted_sentinel_engineering(self):
        frame = pd.DataFrame({'pdays':[999,3], 'previous':[0,2], 'campaign':[1,4]})
        engineered = features(frame,'xgb_engineered')
        self.assertTrue(np.isnan(engineered.days_since_previous_contact.iloc[0]))
        self.assertEqual(engineered.previously_contacted.tolist(),[0,1])
        self.assertNotIn('pdays',engineered)

if __name__ == '__main__':
    unittest.main()
