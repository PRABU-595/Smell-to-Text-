"""
Data augmentation techniques for smell descriptions
"""
import random
import re
from typing import List, Dict, Tuple, Optional
import json


class SmellDataAugmentor:
    """Data augmentation for smell-to-molecule dataset."""
    
    # Synonym dictionaries for smell terms
    SYNONYMS = {
        # Intensity modifiers
        'strong': ['intense', 'powerful', 'potent', 'bold', 'heavy'],
        'light': ['subtle', 'delicate', 'soft', 'faint', 'gentle'],
        'moderate': ['balanced', 'medium', 'mild'],
        
        # Citrus family
        'citrus': ['citrusy', 'zesty', 'tangy'],
        'lemon': ['lemony', 'citron'],
        'orange': ['orangey', 'neroli'],
        'fresh': ['clean', 'crisp', 'airy', 'bright'],
        
        # Floral family
        'floral': ['flowery', 'blooming', 'botanical'],
        'rose': ['rosy', 'rose-like'],
        'jasmine': ['jasmine-like', 'jasmonic'],
        
        # Woody family
        'woody': ['forest-like', 'wood-like', 'tree-like'],
        'cedar': ['cedary', 'cedar-like'],
        'sandalwood': ['creamy wood', 'sandalwood-like'],
        
        # Sweet family
        'sweet': ['sugary', 'honeyed', 'saccharine'],
        'vanilla': ['vanillic', 'vanilla-like', 'creamy'],
        'caramel': ['caramelized', 'toffee-like', 'burnt sugar'],
        
        # Spicy family
        'spicy': ['spiced', 'peppery', 'warm'],
        'cinnamon': ['cinnamony', 'cinnamon-like'],
        
        # Common modifiers
        'warm': ['cozy', 'toasty', 'comforting'],
        'cool': ['cold', 'icy', 'refreshing'],
        'rich': ['deep', 'luxurious', 'opulent'],
        'smooth': ['silky', 'velvety', 'creamy'],
    }
    
    # Sentence templates for reformulation
    TEMPLATES = [
        "A {adj1} fragrance with {notes}",
        "The scent features {notes} with a {adj1} character",
        "{adj1} and {adj2} notes of {notes}",
        "An aroma that combines {notes}",
        "{notes} blend together in this {adj1} composition",
        "This fragrance opens with {notes}",
        "A {adj1} blend of {notes}",
        "Notes of {notes} create a {adj1} impression",
    ]
    
    # Adjectives for descriptions
    ADJECTIVES = [
        'fresh', 'warm', 'soft', 'bright', 'rich', 'smooth', 
        'elegant', 'refined', 'vibrant', 'subtle', 'bold',
        'delicate', 'complex', 'sophisticated', 'natural'
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize augmentor with random seed."""
        self.seed = seed
        random.seed(seed)
    
    def synonym_replacement(self, text: str, n_replacements: int = 2) -> str:
        """
        Replace words with synonyms.
        
        Args:
            text: Original text
            n_replacements: Number of replacements to make
            
        Returns:
            Augmented text
        """
        words = text.lower().split()
        replaceable = [(i, w) for i, w in enumerate(words) if w in self.SYNONYMS]
        
        if not replaceable:
            return text
        
        n_replacements = min(n_replacements, len(replaceable))
        to_replace = random.sample(replaceable, n_replacements)
        
        for idx, word in to_replace:
            synonym = random.choice(self.SYNONYMS[word])
            words[idx] = synonym
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, n_insertions: int = 1) -> str:
        """
        Randomly insert intensity modifiers.
        
        Args:
            text: Original text
            n_insertions: Number of insertions
            
        Returns:
            Augmented text
        """
        modifiers = ['very', 'quite', 'rather', 'slightly', 'somewhat', 'distinctly']
        words = text.split()
        
        for _ in range(n_insertions):
            if len(words) < 2:
                break
            insert_pos = random.randint(1, len(words) - 1)
            modifier = random.choice(modifiers)
            words.insert(insert_pos, modifier)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n_swaps: int = 1) -> str:
        """
        Randomly swap adjacent words.
        
        Args:
            text: Original text
            n_swaps: Number of swaps
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        for _ in range(n_swaps):
            if len(words) < 2:
                break
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def template_reformulation(self, notes: List[str]) -> str:
        """
        Create new description from notes using templates.
        
        Args:
            notes: List of smell notes
            
        Returns:
            New description
        """
        template = random.choice(self.TEMPLATES)
        adj1 = random.choice(self.ADJECTIVES)
        adj2 = random.choice([a for a in self.ADJECTIVES if a != adj1])
        
        if len(notes) == 1:
            notes_str = notes[0]
        elif len(notes) == 2:
            notes_str = f"{notes[0]} and {notes[1]}"
        else:
            notes_str = ", ".join(notes[:-1]) + f", and {notes[-1]}"
        
        result = template.format(adj1=adj1, adj2=adj2, notes=notes_str)
        return result
    
    def augment_description(self, description: str, 
                           techniques: List[str] = None) -> List[str]:
        """
        Apply multiple augmentation techniques.
        
        Args:
            description: Original description
            techniques: List of techniques to apply
            
        Returns:
            List of augmented descriptions
        """
        if techniques is None:
            techniques = ['synonym', 'insertion', 'swap']
        
        augmented = []
        
        if 'synonym' in techniques:
            augmented.append(self.synonym_replacement(description))
        
        if 'insertion' in techniques:
            augmented.append(self.random_insertion(description))
        
        if 'swap' in techniques:
            augmented.append(self.random_swap(description))
        
        return augmented
    
    def create_mixture_sample(self, samples: List[Dict], 
                              n_mix: int = 2) -> Dict:
        """
        Create new sample by mixing chemicals from multiple samples.
        
        Args:
            samples: List of sample dictionaries
            n_mix: Number of samples to mix
            
        Returns:
            New sample dictionary
        """
        selected = random.sample(samples, min(n_mix, len(samples)))
        
        # Combine chemicals
        all_chemicals = []
        all_notes = []
        
        for sample in selected:
            if 'chemicals' in sample:
                for chem in sample['chemicals']:
                    chem_copy = chem.copy()
                    chem_copy['weight'] = chem.get('weight', 1.0) / n_mix
                    all_chemicals.append(chem_copy)
            if 'notes' in sample:
                all_notes.extend(sample['notes'])
        
        # Create new description
        unique_notes = list(set(all_notes))
        new_description = self.template_reformulation(unique_notes[:5])
        
        return {
            'description': new_description,
            'chemicals': all_chemicals,
            'notes': unique_notes,
            'is_synthetic': True
        }
    
    def augment_dataset(self, samples: List[Dict], 
                        augmentation_factor: float = 2.0) -> List[Dict]:
        """
        Augment entire dataset.
        
        Args:
            samples: List of sample dictionaries
            augmentation_factor: How many times to expand dataset
            
        Returns:
            Augmented dataset
        """
        augmented_samples = samples.copy()
        n_augment = int(len(samples) * (augmentation_factor - 1))
        
        for _ in range(n_augment):
            original = random.choice(samples)
            
            technique = random.choice(['synonym', 'template', 'mixture'])
            
            if technique == 'synonym':
                new_desc = self.synonym_replacement(original['description'])
                new_sample = original.copy()
                new_sample['description'] = new_desc
                new_sample['augmentation'] = 'synonym'
                augmented_samples.append(new_sample)
                
            elif technique == 'template' and 'notes' in original:
                new_desc = self.template_reformulation(original['notes'])
                new_sample = original.copy()
                new_sample['description'] = new_desc
                new_sample['augmentation'] = 'template'
                augmented_samples.append(new_sample)
                
            elif technique == 'mixture':
                new_sample = self.create_mixture_sample(samples)
                new_sample['augmentation'] = 'mixture'
                augmented_samples.append(new_sample)
        
        return augmented_samples


def back_translation_augment(text: str, intermediate_lang: str = 'de') -> str:
    """
    Augment text using back-translation (requires translation API).
    
    Args:
        text: Original text
        intermediate_lang: Language code for intermediate translation
        
    Returns:
        Back-translated text
    """
    # Placeholder - would need actual translation API
    # In practice, use Google Translate API or similar
    return text  # Return original as placeholder


if __name__ == '__main__':
    augmentor = SmellDataAugmentor()
    
    # Test augmentation
    test_desc = "A fresh citrus fragrance with light floral notes and a warm woody base"
    
    print(f"Original: {test_desc}\n")
    print("Augmented versions:")
    for aug in augmentor.augment_description(test_desc):
        print(f"  - {aug}")
    
    # Test template reformulation
    test_notes = ['citrus', 'rose', 'sandalwood']
    print(f"\nTemplate reformulation of {test_notes}:")
    for _ in range(3):
        print(f"  - {augmentor.template_reformulation(test_notes)}")
