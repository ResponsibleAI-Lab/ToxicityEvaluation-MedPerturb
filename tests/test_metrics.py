import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate import group_metrics

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.records = json.loads((Path(__file__).resolve().parents[1] / 'examples/scored.json').read_text())

    def test_legacy_metrics_and_threshold_boundary(self):
        result = group_metrics(self.records, 'gender', 0.2, 'legacy')['female']
        self.assertAlmostEqual(result['EMT'], 1.1 / 3)
        self.assertAlmostEqual(result['TF'], 2 / 3)
        self.assertEqual(result['EPBG'], 1)
        self.assertEqual(result['completions'], 3)

    def test_mean_prompt_maxima(self):
        self.assertAlmostEqual(group_metrics(self.records, 'gender', 0.2, 'prompt-max')['female']['EMT'], 0.5)

    def test_unknown_group_and_empty_prompt(self):
        self.records.append({'completions': [{'toxicity_score': None}]})
        result = group_metrics(self.records, 'race', 0.2, 'legacy')
        self.assertEqual(list(result), ['unknown'])
        self.assertEqual(result['unknown']['prompts'], 2)

    def test_invalid_scores(self):
        self.records[0]['completions'][0]['toxicity_score'] = float('nan')
        with self.assertRaises(ValueError):
            group_metrics(self.records, 'age', 0.2, 'legacy')

if __name__ == '__main__':
    unittest.main()
