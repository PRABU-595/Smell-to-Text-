import json
from pathlib import Path

# Paths
json_path = Path('data/processed/family_chemicals.json')

with open(json_path, 'r') as f:
    data = json.load(f)

# The "Biological Malodor/Musk" expert injection
new_chems = [
    {
        "family": "malodor_biological",
        "chems": [
            {"cas": "503-74-2", "name": "Isovaleric acid", "descriptors": "sweaty;cheesy;acrid;sour;feet"},
            {"cas": "27960-21-0", "name": "3-Methyl-2-hexenoic acid", "descriptors": "sweaty;underarm;acrid;biological"},
            {"cas": "64-18-6", "name": "Formic acid", "descriptors": "acrid;sour;stinging;sweaty"}
        ]
    },
    {
        "family": "musky_animalic",
        "chems": [
            {"cas": "18330-05-3", "name": "5-alpha-Androst-16-en-3-one", "descriptors": "musky;animalic;pheromonal;sweaty;urine"}
        ]
    }
]

for entry in new_chems:
    fam = entry['family']
    if fam not in data:
        data[fam] = []
    
    existing_cas = [c.get('cas') for c in data[fam]]
    for chem in entry['chems']:
        if chem['cas'] not in existing_cas:
            data[fam].append(chem)

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Malodor Expert Knowledge Injected: Biological markers added.")
