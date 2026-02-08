"""Tests for utility modules."""
import sys
sys.path.append('.')

import pytest
import tempfile
import json
from pathlib import Path


class TestConfigLoader:
    def test_import(self):
        from src.utils.config_loader import ConfigLoader
        loader = ConfigLoader()
        assert loader is not None
    
    def test_default_config(self):
        from src.utils.config_loader import get_default_config
        config = get_default_config()
        assert 'model' in config
        assert 'training' in config


class TestLogger:
    def test_setup_logger(self):
        from src.utils.logger import setup_logger
        logger = setup_logger('test')
        assert logger is not None
    
    def test_training_logger(self):
        from src.utils.logger import setup_logger, TrainingLogger
        base_logger = setup_logger('test')
        training_logger = TrainingLogger(base_logger)
        training_logger.log_epoch(1, {'loss': 0.5})
        assert len(training_logger.history) == 1


class TestHelpers:
    def test_set_seed(self):
        from src.utils.helpers import set_seed
        set_seed(42)
    
    def test_ensure_dir(self):
        from src.utils.helpers import ensure_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            path = ensure_dir(f"{tmpdir}/test/subdir")
            assert path.exists()
    
    def test_save_load_json(self):
        from src.utils.helpers import save_json, load_json
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            data = {'key': 'value', 'number': 42}
            save_json(data, f.name)
            loaded = load_json(f.name)
            assert loaded == data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
