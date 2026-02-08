"""
Text preprocessing utilities
"""
import re
import string

class TextCleaner:
    def __init__(self):
        self.punctuation = string.punctuation
    
    def clean_description(self, text):
        """Clean smell description text"""
        # Lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def normalize_chemical_name(self, name):
        """Normalize chemical names"""
        name = name.strip().lower()
        # Remove CAS numbers, brand names, etc.
        name = re.sub(r'\(\d+\-\d+\-\d+\)', '', name)
        return name.strip()
