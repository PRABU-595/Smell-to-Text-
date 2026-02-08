"""Tests for data preprocessing modules."""
import sys
sys.path.append('.')

import pytest
from src.data.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()
    
    def test_clean_basic(self):
        text = "  Hello   World!  "
        result = self.cleaner.clean_description(text)
        assert "hello" in result.lower()
    
    def test_lowercase(self):
        result = self.cleaner.clean_description("UPPERCASE TEXT")
        assert result.islower() or result == result.lower()
    
    def test_remove_extra_spaces(self):
        result = self.cleaner.clean_description("too    many   spaces")
        assert "  " not in result
    
    def test_empty_string(self):
        result = self.cleaner.clean_description("")
        assert result == ""
    
    def test_special_characters(self):
        result = self.cleaner.clean_description("text@#$%special")
        assert "@" not in result


class TestChemicalMapper:
    def test_import(self):
        from src.data.preprocessing.chemical_mapper import ChemicalMapper
        mapper = ChemicalMapper()
        assert mapper is not None


class TestDataAugmentation:
    def test_import(self):
        from src.data.preprocessing.data_augmentation import SmellDataAugmentor
        augmentor = SmellDataAugmentor()
        assert augmentor is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
