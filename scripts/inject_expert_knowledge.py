import json
from pathlib import Path

# Paths
json_path = Path('data/processed/family_chemicals.json')

# Load
with open(json_path, 'r') as f:
    data = json.load(f)

# The "Earthy" expert injection
geosmin = {
    "cas": "19700-21-1",
    "name": "Geosmin",
    "descriptors": "earthy;musty;rain;damp soil;petrichor;beet;moldy"
}
mib = {
    "cas": "2371-42-8",
    "name": "2-Methylisoborneol",
    "descriptors": "earthy;musty;camphoreous;muddy;rain"
}

# Add to Earthy (or create it if it didn't exist)
target_family = "earthy"
if target_family not in data:
    data[target_family] = []

# Avoid duplicates
existing_cas = [c.get('cas') for c in data[target_family]]
if geosmin['cas'] not in existing_cas:
    data[target_family].append(geosmin)
if mib['cas'] not in existing_cas:
    data[target_family].append(mib)

# Save back
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Expert Knowledge Injected: Geosmin and 2-MIB added to '{target_family}' family.")
