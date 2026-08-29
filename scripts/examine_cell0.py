import json

with open('notebooks/04_data_analysis/04_01_eda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][0]
print("Source:")
print(''.join(cell['source']))
