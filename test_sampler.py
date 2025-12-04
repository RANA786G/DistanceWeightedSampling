import unittest
from unittest.mock import patch
from sampler import BalancedBatchSampler
import torch

# A simple mock dataset
class MockDataset:
    def __init__(self):
        self.train_labels = torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    def __len__(self):
        return len(self.train_labels)

class TestSampler(unittest.TestCase):
    # We patch _get_label so we don't have to worry about the type check in the original implementation.
    # The purpose of this test is to verify that the sampler *terminates*, not to test the label extraction.
    @patch('sampler.BalancedBatchSampler._get_label', new=lambda self, dataset, idx: dataset.train_labels[idx].item())
    def test_sampler_finishes(self):
        dataset = MockDataset()
        sampler = BalancedBatchSampler(dataset, batch_size=4, batch_k=2, length=10)

        count = 0
        for batch in sampler:
            count += 1
        self.assertEqual(count, 10)

if __name__ == '__main__':
    unittest.main()
