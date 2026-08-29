import json, glob

for f in sorted(glob.glob('notebooks/04_data_analysis/04_*.ipynb')):
    with open(f, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    total_lines = len(json.dumps(nb, indent=1).split('\n'))
    print(f'{f.split("/")[-1]}: {len(nb["cells"])} cells ({len(md_cells)} md, {len(code_cells)} code) ~{total_lines} lines')
