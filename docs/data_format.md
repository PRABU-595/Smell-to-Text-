# Data Format

## Processed Data Format

The processed data is stored as CSV files with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| description | string | Cleaned smell description |
| chemicals | json array | List of associated chemical compounds |

### Example
```csv
description,chemicals
"Fresh citrus bergamot lemon","[""Limonene"", ""Citral"", ""Linalool""]"
"Warm woody sandalwood","[""Santalol"", ""Cedrene""]"
```

## Chemical Mapping Format

Chemical mappings are stored as JSON:

```json
{
  "chemical_name": {
    "id": "pubchem_id",
    "smiles": "SMILES_string",
    "formula": "molecular_formula",
    "categories": ["citrus", "fresh"]
  }
}
```

## Label Encoding

Chemicals are encoded as multi-hot vectors:
- Shape: `(num_samples, num_chemicals)`
- Values: 0 or 1
- Order matches `chemical_vocabulary.json`
