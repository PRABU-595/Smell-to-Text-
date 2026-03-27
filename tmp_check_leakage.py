import pandas as pd

train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Check description overlap
train_descs = set(train['description'].str.lower().str.strip())
test_descs = set(test['description'].str.lower().str.strip())

exact_overlap = train_descs & test_descs
print(f"\nExact description overlap train/test: {len(exact_overlap)}")

# Check if augmented variants share a root description
def extract_root(desc):
    desc = desc.lower().strip()
    for prefix in ['a scent characterized by ', 'the aroma profile is primarily ',
                   'notes of ', 'this fragrant molecule features ',
                   'this aromatic molecule features ', 'this distinct molecule features ',
                   'this notable molecule features ', 'this clear molecule features ']:
        if desc.startswith(prefix):
            desc = desc[len(prefix):]
    for suffix in ['.', ', with']:
        if suffix in desc:
            desc = desc.split(suffix)[0]
    desc = desc.rstrip(' .')
    return desc

train['root'] = train['description'].apply(extract_root)
test['root'] = test['description'].apply(extract_root)

train_roots = set(train['root'])
test_roots = set(test['root'])
root_overlap = train_roots & test_roots

print(f"Root description overlap train/test: {len(root_overlap)} / {len(test_roots)} test roots")
pct = len(root_overlap) / len(test_roots) * 100
print(f"Percentage of test set with leaked roots: {pct:.1f}%")

if root_overlap:
    print("\nSample leaked roots:")
    for r in list(root_overlap)[:5]:
        print(f"  - '{r}'")
