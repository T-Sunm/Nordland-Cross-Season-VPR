import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from nordland_vpr import (BoQ, PairBatchSampler, folder_table, make_mock_test,
                          predict_folder, retrieval_metrics, score_like_competition,
                          seed_everything, topk_search, tutorial_metadata,
                          validate_submission)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        seed_everything(12)
        torch.set_num_threads(2)

    def test_boq_forward_backward(self):
        model = BoQ(in_channels=16, proj_channels=32, num_queries=4,
                    num_layers=2, output_dim=128, nheads=4)
        x = torch.randn(2, 16, 4, 4, requires_grad=True)
        descriptor, weights = model(x, return_attention=True)
        self.assertEqual(tuple(descriptor.shape), (2, 128))
        self.assertEqual(len(weights), 2)
        self.assertEqual(tuple(weights[0].shape), (2, 4, 16))
        self.assertTrue(torch.allclose(descriptor.norm(dim=1), torch.ones(2), atol=1e-5))
        (descriptor[0] * descriptor[1]).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.boqs[0].queries.grad)

    def test_pair_sampler(self):
        table = pd.DataFrame({'position':[1, 1, 2, 2, 3, 3],
                              'season':['summer','winter'] * 3})
        sampler = PairBatchSampler(table, places_per_batch=2, seed=2)
        for indices in sampler:
            subset = table.iloc[indices]
            self.assertTrue((subset.position.value_counts() == 2).all())

    def test_search_and_metrics(self):
        queries = np.eye(3, dtype=np.float32)
        idx, score = topk_search(queries, queries, k=3)
        self.assertTrue(np.array_equal(idx[:, 0], [0, 1, 2]))
        metric, _ = retrieval_metrics(np.array([100,200,300]),
                                      np.array([105,199,300]), idx, threshold=10)
        self.assertEqual(metric['accuracy_pm10'], 100)

    def test_submission(self):
        frame = pd.DataFrame({'query_id':['q1', 'q2'], 'winter_id':['w1', 'w1']})
        self.assertTrue(validate_submission(frame, ['q1','q2'], ['w1', 'w2']))
        with self.assertRaises(ValueError):
            validate_submission(pd.concat([frame, frame.iloc[:1]], ignore_index=True),
                                ['q1','q2'], ['w1','w2'])
        actual = pd.DataFrame([{'query_id':'q1','winter_id':'w1'},
                               {'query_id':'q1','winter_id':'w2'},
                               {'query_id':'q2','winter_id':'missing'}])
        score = score_like_competition(actual, ['q1','q2'], ['w1','w2'],
                                       {'q1':100,'q2':200}, {'w1':101,'w2':200})
        self.assertEqual(score, 50.0)

    def test_tutorial_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for position in (251,1640,1641,1650,1651,2000):
                for season in ('summer','winter'):
                    folder = root / season
                    folder.mkdir(exist_ok=True)
                    Image.new('RGB', (10,10), 'red').save(folder / f'images-{position:05}.png')
            metadata = tutorial_metadata(root, require_complete=False)
            self.assertEqual(metadata.groupby('split').size().to_dict(),
                             {'train':4, 'buffer':4, 'validation':4})
            mock = make_mock_test(metadata, root/'artifacts', seed=5, query_stride=1)
            self.assertEqual(len(list((mock/'queries').glob('*.jpg'))), 2)
            self.assertEqual(len(list((mock/'database').glob('*.jpg'))), 2)


if __name__ == '__main__':
    unittest.main()
