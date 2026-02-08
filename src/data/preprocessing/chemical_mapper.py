"""
Chemical mapping utilities for smell-to-molecule translation
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict


class ChemicalMapper:
    """Maps smell descriptors to chemical compounds."""
    
    # Common smell note to chemical mappings
    DEFAULT_MAPPINGS = {
        # Citrus notes
        'citrus': ['Limonene', 'Citral', 'Linalyl acetate'],
        'lemon': ['Limonene', 'Citral', 'Neral'],
        'orange': ['Limonene', 'Linalool', 'Myrcene'],
        'bergamot': ['Linalyl acetate', 'Limonene', 'Linalool'],
        'grapefruit': ['Nootkatone', 'Limonene', 'Myrcene'],
        'lime': ['Limonene', 'Citral', 'Gamma-terpinene'],
        
        # Floral notes
        'floral': ['Linalool', 'Geraniol', 'Phenylethyl alcohol'],
        'rose': ['Geraniol', 'Citronellol', 'Phenylethyl alcohol'],
        'jasmine': ['Benzyl acetate', 'Jasmone', 'Indole'],
        'lavender': ['Linalool', 'Linalyl acetate', 'Camphor'],
        'violet': ['Ionone', 'Methyl ionone'],
        'iris': ['Ionone', 'Orris root'],
        'lily': ['Hydroxycitronellal', 'Linalool'],
        
        # Woody notes
        'woody': ['Cedrene', 'Santalol', 'Vetiverol'],
        'cedar': ['Cedrene', 'Thujopsene', 'Cedrol'],
        'sandalwood': ['Santalol', 'Santene'],
        'pine': ['Pinene', 'Limonene', 'Bornyl acetate'],
        'vetiver': ['Vetiverol', 'Khusimol'],
        'oud': ['Agarospirol', 'Jinkohol'],
        
        # Sweet notes
        'vanilla': ['Vanillin', 'Ethyl vanillin'],
        'caramel': ['Furaneol', 'Maltol', 'Cyclotene'],
        'honey': ['Phenylacetic acid', 'Methyl phenylacetate'],
        'chocolate': ['Pyrazines', 'Methylbutanal'],
        
        # Fresh notes
        'fresh': ['Linalool', 'Dihydromyrcenol', 'Calone'],
        'aquatic': ['Calone', 'Dihydromyrcenol'],
        'marine': ['Calone', 'Helional'],
        'ozonic': ['Calone', 'Floralozone'],
        
        # Spicy notes
        'spicy': ['Eugenol', 'Cinnamaldehyde', 'Carvone'],
        'cinnamon': ['Cinnamaldehyde', 'Eugenol'],
        'clove': ['Eugenol', 'Beta-caryophyllene'],
        'pepper': ['Piperine', 'Rotundone'],
        'ginger': ['Gingerol', 'Zingiberene'],
        
        # Herbal notes
        'herbal': ['Linalool', 'Thymol', 'Carvacrol'],
        'mint': ['Menthol', 'Menthone', 'Menthyl acetate'],
        'basil': ['Linalool', 'Estragole', 'Eugenol'],
        'thyme': ['Thymol', 'Carvacrol'],
        'rosemary': ['Camphor', 'Cineole', 'Pinene'],
        
        # Musky notes
        'musk': ['Muscone', 'Galaxolide', 'Ambrettolide'],
        'amber': ['Ambroxide', 'Labdanum'],
        'powdery': ['Heliotropin', 'Ionone', 'Coumarin'],
        
        # Fruity notes
        'fruity': ['Ethyl butyrate', 'Isoamyl acetate', 'Gamma-decalactone'],
        'apple': ['Ethyl-2-methylbutyrate', 'Hexyl acetate'],
        'peach': ['Gamma-decalactone', 'Gamma-undecalactone'],
        'berry': ['Ethyl butyrate', 'Furaneol'],
        'tropical': ['Allyl hexanoate', 'Ethyl butyrate'],
    }
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize chemical mapper.
        
        Args:
            mapping_file: Path to JSON file with custom mappings
        """
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        self.reverse_mappings = self._build_reverse_mappings()
        
        if mapping_file and Path(mapping_file).exists():
            self.load_mappings(mapping_file)
    
    def _build_reverse_mappings(self) -> Dict[str, List[str]]:
        """Build reverse mappings (chemical -> descriptors)."""
        reverse = defaultdict(list)
        for descriptor, chemicals in self.mappings.items():
            for chem in chemicals:
                reverse[chem.lower()].append(descriptor)
        return dict(reverse)
    
    def load_mappings(self, filepath: str) -> None:
        """Load custom mappings from JSON file."""
        with open(filepath, 'r') as f:
            custom_mappings = json.load(f)
        self.mappings.update(custom_mappings)
        self.reverse_mappings = self._build_reverse_mappings()
    
    def save_mappings(self, filepath: str) -> None:
        """Save mappings to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.mappings, f, indent=2)
    
    def get_chemicals(self, descriptor: str) -> List[str]:
        """
        Get chemicals associated with a smell descriptor.
        
        Args:
            descriptor: Smell descriptor (e.g., 'citrus', 'woody')
            
        Returns:
            List of chemical names
        """
        descriptor = descriptor.lower().strip()
        return self.mappings.get(descriptor, [])
    
    def get_descriptors(self, chemical: str) -> List[str]:
        """
        Get smell descriptors associated with a chemical.
        
        Args:
            chemical: Chemical name
            
        Returns:
            List of descriptors
        """
        chemical = chemical.lower().strip()
        return self.reverse_mappings.get(chemical, [])
    
    def map_description(self, description: str) -> Dict[str, float]:
        """
        Map a smell description to chemicals with weights.
        
        Args:
            description: Natural language smell description
            
        Returns:
            Dictionary of chemical -> weight mappings
        """
        description = description.lower()
        chemical_counts = defaultdict(int)
        
        # Find all matching descriptors
        for descriptor in self.mappings:
            if descriptor in description:
                for chemical in self.mappings[descriptor]:
                    chemical_counts[chemical] += 1
        
        if not chemical_counts:
            return {}
        
        # Normalize to weights
        total = sum(chemical_counts.values())
        weights = {chem: count / total for chem, count in chemical_counts.items()}
        
        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    def extract_notes(self, description: str) -> List[str]:
        """
        Extract smell notes from description.
        
        Args:
            description: Natural language description
            
        Returns:
            List of extracted notes
        """
        description = description.lower()
        found_notes = []
        
        for note in self.mappings:
            if note in description:
                found_notes.append(note)
        
        return found_notes
    
    def create_training_sample(self, description: str, 
                                chemicals: List[Dict]) -> Dict:
        """
        Create a training sample from description and chemicals.
        
        Args:
            description: Smell description
            chemicals: List of {name, formula, weight} dicts
            
        Returns:
            Training sample dictionary
        """
        return {
            'description': description,
            'chemicals': chemicals,
            'notes': self.extract_notes(description)
        }
    
    def add_mapping(self, descriptor: str, chemicals: List[str]) -> None:
        """Add or update a descriptor mapping."""
        self.mappings[descriptor.lower()] = chemicals
        self.reverse_mappings = self._build_reverse_mappings()
    
    def get_all_chemicals(self) -> List[str]:
        """Get list of all known chemicals."""
        all_chemicals = set()
        for chemicals in self.mappings.values():
            all_chemicals.update(chemicals)
        return sorted(list(all_chemicals))
    
    def get_all_descriptors(self) -> List[str]:
        """Get list of all known descriptors."""
        return sorted(list(self.mappings.keys()))


def build_chemical_vocabulary(data_path: str, output_path: str) -> Dict[str, int]:
    """
    Build chemical vocabulary from dataset.
    
    Args:
        data_path: Path to processed data CSV
        output_path: Path to save vocabulary JSON
        
    Returns:
        Dictionary mapping chemical names to indices
    """
    df = pd.read_csv(data_path)
    
    all_chemicals = set()
    for chemicals_str in df['chemicals']:
        chemicals = json.loads(chemicals_str)
        for chem in chemicals:
            all_chemicals.add(chem['name'])
    
    vocab = {chem: idx for idx, chem in enumerate(sorted(all_chemicals))}
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    return vocab


if __name__ == '__main__':
    # Test the mapper
    mapper = ChemicalMapper()
    
    test_descriptions = [
        "Fresh citrus with floral notes",
        "Warm, woody, slightly sweet with sandalwood",
        "Sweet vanilla with hints of caramel and tonka"
    ]
    
    for desc in test_descriptions:
        print(f"\nDescription: {desc}")
        print(f"Notes: {mapper.extract_notes(desc)}")
        print(f"Chemicals: {mapper.map_description(desc)}")
