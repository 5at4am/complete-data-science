import json

with open("notebooks/04_data_analysis/04_01_eda.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][-1]
print(f"=== Cell {len(nb['cells']) - 1} (last) ===")
print("".join(cell["source"]))
