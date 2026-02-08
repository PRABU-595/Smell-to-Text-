"""
Tokenization utilities for smell descriptions
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json


class SmellTokenizer:
    """Custom tokenizer for smell descriptions with domain-specific handling."""
    
    # Special tokens
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    
    # Domain-specific tokens for smell notes
    SMELL_TOKENS = [
        '[CITRUS]', '[FLORAL]', '[WOODY]', '[SPICY]', '[SWEET]',
        '[FRESH]', '[MUSKY]', '[HERBAL]', '[FRUITY]', '[AQUATIC]',
        '[POWDERY]', '[SMOKY]', '[EARTHY]', '[GREEN]', '[ORIENTAL]'
    ]
    
    def __init__(self, vocab_file: Optional[str] = None, 
                 max_vocab_size: int = 10000):
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Path to vocabulary JSON file
            max_vocab_size: Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        # Initialize special tokens
        self._init_special_tokens()
        
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def _init_special_tokens(self) -> None:
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN,
            self.SEP_TOKEN, self.MASK_TOKEN
        ] + self.SMELL_TOKENS
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from corpus.
        
        Args:
            texts: List of text documents
            min_freq: Minimum frequency for word inclusion
        """
        word_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Add words above frequency threshold
        current_idx = len(self.word2idx)
        for word, count in word_counts.most_common():
            if count < min_freq:
                break
            if word not in self.word2idx and current_idx < self.max_vocab_size:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Handle special smell terms
        text = self._replace_smell_terms(text)
        
        # Basic word tokenization
        # Keep hyphens in compound words
        tokens = re.findall(r'\[[\w]+\]|[\w]+-[\w]+|[\w]+', text)
        
        return tokens
    
    def _replace_smell_terms(self, text: str) -> str:
        """Replace smell category terms with special tokens."""
        replacements = {
            r'\b(citrus|citrusy|lemon|lime|orange|bergamot|grapefruit)\b': '[CITRUS]',
            r'\b(floral|flower|rose|jasmine|lily|violet|iris)\b': '[FLORAL]',
            r'\b(woody|wood|cedar|sandalwood|vetiver|oud)\b': '[WOODY]',
            r'\b(spicy|spice|pepper|cinnamon|clove|cardamom)\b': '[SPICY]',
            r'\b(sweet|sugary|vanilla|caramel|honey|toffee)\b': '[SWEET]',
            r'\b(fresh|clean|crisp|airy|bright)\b': '[FRESH]',
            r'\b(musky|musk|amber|sensual)\b': '[MUSKY]',
            r'\b(herbal|herb|mint|basil|thyme|lavender)\b': '[HERBAL]',
            r'\b(fruity|fruit|apple|peach|berry|tropical)\b': '[FRUITY]',
            r'\b(aquatic|marine|oceanic|watery|ozonic)\b': '[AQUATIC]',
        }
        
        for pattern, token in replacements.items():
            text = re.sub(pattern, token, text, flags=re.IGNORECASE)
        
        return text
    
    def encode(self, text: str, max_length: int = 128,
               add_special_tokens: bool = True) -> Dict:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_special_tokens: Whether to add [CLS] and [SEP]
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        
        # Convert to IDs
        input_ids = []
        for token in tokens[:max_length]:
            if token in self.word2idx:
                input_ids.append(self.word2idx[token])
            else:
                input_ids.append(self.word2idx[self.UNK_TOKEN])
        
        # Padding
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        
        if padding_length > 0:
            input_ids += [self.word2idx[self.PAD_TOKEN]] * padding_length
            attention_mask += [0] * padding_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens[:max_length]
        }
    
    def decode(self, input_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            input_ids: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        special = {self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN}
        
        for idx in input_ids:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special and token in special:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)
        
        return ' '.join(tokens)
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f, indent=2)
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def get_special_tokens_mask(self, input_ids: List[int]) -> List[int]:
        """Get mask indicating special tokens."""
        special_ids = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.CLS_TOKEN],
            self.word2idx[self.SEP_TOKEN]
        }
        return [1 if idx in special_ids else 0 for idx in input_ids]


class ChemicalTokenizer:
    """Tokenizer for chemical formulas and SMILES."""
    
    def __init__(self):
        """Initialize chemical tokenizer."""
        self.atom_pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    
    def tokenize_formula(self, formula: str) -> List[str]:
        """
        Tokenize molecular formula.
        
        Args:
            formula: Molecular formula (e.g., "C10H16O")
            
        Returns:
            List of tokens
        """
        tokens = []
        matches = self.atom_pattern.findall(formula)
        
        for atom, count in matches:
            if atom:
                tokens.append(atom)
                if count:
                    tokens.append(count)
        
        return tokens
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string.
        
        Args:
            smiles: SMILES notation
            
        Returns:
            List of tokens
        """
        # SMILES tokenization pattern
        pattern = r'(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|@|[=#$:\\\/]|[CNOSPFIBcnops]|\d)'
        tokens = re.findall(pattern, smiles)
        return tokens


if __name__ == '__main__':
    # Test smell tokenizer
    tokenizer = SmellTokenizer()
    
    test_texts = [
        "Fresh citrus with bergamot and lemon notes",
        "Warm, woody fragrance with sandalwood base",
        "Sweet vanilla and caramel with floral hints"
    ]
    
    # Build vocabulary
    tokenizer.build_vocab(test_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encoding
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {encoded['tokens']}")
        print(f"IDs: {encoded['input_ids'][:10]}...")
    
    # Test chemical tokenizer
    chem_tokenizer = ChemicalTokenizer()
    
    print("\n\nChemical tokenization:")
    formulas = ["C10H16", "C10H18O", "C8H8O3"]
    for formula in formulas:
        tokens = chem_tokenizer.tokenize_formula(formula)
        print(f"  {formula} -> {tokens}")
