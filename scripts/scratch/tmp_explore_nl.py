import pandas as pd

# 1. Perfume Recommendation
try:
    df = pd.read_csv('data/raw/natural_language/perfume_recommendation/final_perfume_data.csv', encoding='latin-1')
    print('=== PERFUME RECOMMENDATION ===')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print()
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"  [{i}] Name: {row.get('Name', 'N/A')}")
        desc = str(row.get('Description', ''))[:150]
        print(f"      Desc: {desc}")
        notes = str(row.get('Notes', ''))[:100]
        print(f"      Notes: {notes}")
        print()
except Exception as e:
    print(f"Perfume CSV error: {e}")

# 2. Laymen free descriptions
try:
    df2 = pd.read_excel('data/raw/natural_language/laymen_olfactory/free_descriptions_translated.xlsx')
    print('=== LAYMEN FREE DESCRIPTIONS ===')
    print(f'Shape: {df2.shape}')
    print(f'Columns: {list(df2.columns)[:10]}')
    print()
    for i in range(min(5, len(df2))):
        row = df2.iloc[i]
        print(f"  [{i}] {row.to_dict()}")
        print()
except Exception as e:
    print(f"Laymen XLSX error: {e}")

# 3. Odors list
try:
    df3 = pd.read_excel('data/raw/natural_language/laymen_olfactory/odors.xlsx')
    print('=== ODORS LIST ===')
    print(f'Shape: {df3.shape}')
    print(f'Columns: {list(df3.columns)}')
    print(df3.head(5).to_string())
except Exception as e:
    print(f"Odors error: {e}")
