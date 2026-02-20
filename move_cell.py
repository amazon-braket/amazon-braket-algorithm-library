import json

with open("notebooks/advanced_algorithms/HHL_Algorithm.ipynb", "r") as f:
    nb = json.load(f)

# Find "Understanding the Circuit Components" cell
idx_to_move = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "## Understanding the Circuit Components" in source:
            idx_to_move = i
            break

if idx_to_move != -1:
    cell = nb["cells"].pop(idx_to_move)
    nb["cells"].insert(1, cell)

    with open("notebooks/advanced_algorithms/HHL_Algorithm.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Moved cell successfully.")
else:
    print("Cell not found.")
