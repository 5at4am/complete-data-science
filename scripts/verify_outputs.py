import json, glob

for f in sorted(glob.glob('notebooks/04_data_analysis/04_*.ipynb')):
    with open(f, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    cells_with_output = sum(1 for c in code_cells if c.get('outputs'))
    total_outputs = sum(len(c.get('outputs', [])) for c in code_cells)
    # Check for base64 images
    has_images = any(
        any('image/png' in str(o) for o in c.get('outputs', []))
        for c in code_cells
    )
    name = f.split('/')[-1].split('\\')[-1]
    print(f'{name}: {cells_with_output}/{len(code_cells)} code cells have outputs, {total_outputs} total outputs, images: {has_images}')
