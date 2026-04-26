import json
nb = json.load(open('train.ipynb', 'r', encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"Total: {len(nb['cells'])} | Code: {len(code_cells)} | Markdown: {len(md_cells)}")
print(f"GPU: {nb['metadata']['colab']['gpuType']}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'][:1]).strip()[:60]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {len(c['source']):3d} lines | {src}")
