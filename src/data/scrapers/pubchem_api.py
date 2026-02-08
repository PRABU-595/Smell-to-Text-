"""
PubChem API client for chemical structure retrieval
"""
import requests
import time
from typing import Dict, List, Optional, Union
import json
import pandas as pd
from tqdm import tqdm


class PubChemAPI:
    """Client for PubChem REST API."""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, delay: float = 0.2):
        """
        Initialize PubChem API client.
        
        Args:
            delay: Time to wait between requests (seconds)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
    
    def get_compound_by_name(self, name: str) -> Optional[Dict]:
        """
        Get compound information by chemical name.
        
        Args:
            name: Chemical name (e.g., "Limonene")
            
        Returns:
            Dictionary with compound data or None
        """
        url = f"{self.BASE_URL}/compound/name/{name}/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_compound(data)
            return None
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching {name}: {e}")
            return None
        finally:
            time.sleep(self.delay)
    
    def get_compound_by_cid(self, cid: int) -> Optional[Dict]:
        """
        Get compound information by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            Dictionary with compound data or None
        """
        url = f"{self.BASE_URL}/compound/cid/{cid}/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_compound(data)
            return None
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching CID {cid}: {e}")
            return None
        finally:
            time.sleep(self.delay)
    
    def get_compound_by_cas(self, cas: str) -> Optional[Dict]:
        """
        Get compound information by CAS registry number.
        
        Args:
            cas: CAS registry number (e.g., "5989-27-5")
            
        Returns:
            Dictionary with compound data or None
        """
        # Search by CAS number
        search_url = f"{self.BASE_URL}/compound/name/{cas}/cids/JSON"
        try:
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                cids = data.get('IdentifierList', {}).get('CID', [])
                if cids:
                    return self.get_compound_by_cid(cids[0])
            return None
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching CAS {cas}: {e}")
            return None
        finally:
            time.sleep(self.delay)
    
    def get_smiles(self, compound_id: Union[str, int], id_type: str = 'name') -> Optional[str]:
        """
        Get SMILES notation for a compound.
        
        Args:
            compound_id: Compound identifier
            id_type: Type of identifier ('name', 'cid', 'cas')
            
        Returns:
            SMILES string or None
        """
        url = f"{self.BASE_URL}/compound/{id_type}/{compound_id}/property/CanonicalSMILES/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    return props[0].get('CanonicalSMILES')
            return None
        except (requests.RequestException, json.JSONDecodeError):
            return None
        finally:
            time.sleep(self.delay)
    
    def get_molecular_formula(self, compound_id: Union[str, int], id_type: str = 'name') -> Optional[str]:
        """Get molecular formula for a compound."""
        url = f"{self.BASE_URL}/compound/{id_type}/{compound_id}/property/MolecularFormula/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    return props[0].get('MolecularFormula')
            return None
        except (requests.RequestException, json.JSONDecodeError):
            return None
        finally:
            time.sleep(self.delay)
    
    def get_properties(self, compound_id: Union[str, int], 
                       properties: List[str], id_type: str = 'name') -> Optional[Dict]:
        """
        Get multiple properties for a compound.
        
        Args:
            compound_id: Compound identifier
            properties: List of property names
            id_type: Type of identifier
            
        Returns:
            Dictionary with properties or None
        """
        props_str = ','.join(properties)
        url = f"{self.BASE_URL}/compound/{id_type}/{compound_id}/property/{props_str}/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    return props[0]
            return None
        except (requests.RequestException, json.JSONDecodeError):
            return None
        finally:
            time.sleep(self.delay)
    
    def _parse_compound(self, data: Dict) -> Dict:
        """Parse compound data from API response."""
        compounds = data.get('PC_Compounds', [])
        if not compounds:
            return {}
        
        compound = compounds[0]
        result = {
            'cid': compound.get('id', {}).get('id', {}).get('cid'),
            'atoms': [],
            'bonds': [],
            'properties': {}
        }
        
        # Extract atoms
        if 'atoms' in compound:
            atoms = compound['atoms']
            result['atoms'] = atoms.get('element', [])
        
        # Extract properties
        if 'props' in compound:
            for prop in compound['props']:
                urn = prop.get('urn', {})
                label = urn.get('label', '')
                value = prop.get('value', {})
                
                if 'sval' in value:
                    result['properties'][label] = value['sval']
                elif 'ival' in value:
                    result['properties'][label] = value['ival']
                elif 'fval' in value:
                    result['properties'][label] = value['fval']
        
        return result
    
    def batch_lookup(self, names: List[str], output_file: str = None) -> pd.DataFrame:
        """
        Look up multiple compounds by name.
        
        Args:
            names: List of chemical names
            output_file: Optional CSV file to save results
            
        Returns:
            DataFrame with compound data
        """
        results = []
        
        for name in tqdm(names, desc="Looking up compounds"):
            props = self.get_properties(
                name, 
                ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 'IUPACName'],
                id_type='name'
            )
            if props:
                props['query_name'] = name
                results.append(props)
        
        df = pd.DataFrame(results)
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df


if __name__ == '__main__':
    api = PubChemAPI()
    
    # Test with common fragrance chemicals
    test_chemicals = ['Limonene', 'Linalool', 'Vanillin', 'Coumarin', 'Citral']
    
    for chem in test_chemicals:
        print(f"\n{chem}:")
        formula = api.get_molecular_formula(chem)
        smiles = api.get_smiles(chem)
        print(f"  Formula: {formula}")
        print(f"  SMILES: {smiles}")
