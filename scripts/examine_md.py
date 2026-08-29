import json

with open('notebooks/04_data_analysis/04_01_eda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        first_line = cell['source'][0].strip() if cell['source'] else '(empty)'
        preview = first_line[:80]
        print(f"Cell {i}: {preview}")
