"""Tests for evaluation metrics."""
import sys
sys.path.append('.')

import pytest
import numpy as np
from src.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    def setup_method(self):
        self.calculator = MetricsCalculator()
        np.random.seed(42)
        self.predictions = np.random.rand(10, 50)
        self.labels = np.random.randint(0, 2, (10, 50)).astype(float)
    
    def test_compute_all_metrics(self):
        results = self.calculator.compute_all_metrics(self.predictions, self.labels)
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_precision_at_k(self):
        result = self.calculator.precision_at_k(self.predictions, self.labels, k=5)
        assert 0 <= result <= 1
    
    def test_recall_at_k(self):
        result = self.calculator.recall_at_k(self.predictions, self.labels, k=5)
        assert 0 <= result <= 1
    
    def test_map_score(self):
        result = self.calculator.mean_average_precision(self.predictions, self.labels)
        assert 0 <= result <= 1
    
    def test_ndcg_score(self):
        result = self.calculator.ndcg_score(self.predictions, self.labels, k=10)
        assert 0 <= result <= 1
    
    def test_f1_score(self):
        binary_preds = (self.predictions > 0.5).astype(float)
        result = self.calculator.f1_score_multilabel(binary_preds, self.labels)
        assert 0 <= result <= 1


class TestErrorAnalysis:
    def test_import(self):
        from src.evaluation.error_analysis import ErrorAnalyzer
        analyzer = ErrorAnalyzer([], [])
        assert analyzer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
