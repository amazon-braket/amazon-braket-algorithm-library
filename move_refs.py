import json

with open("notebooks/advanced_algorithms/HHL_Algorithm.ipynb", "r") as f:
    nb = json.load(f)

# Find "References" cell
idx_to_move = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "## References" in source:
            idx_to_move = i
            break

if idx_to_move != -1:
    cell = nb["cells"].pop(idx_to_move)
    nb["cells"].append(cell)

    with open("notebooks/advanced_algorithms/HHL_Algorithm.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Moved References cell to bottom successfully.")
else:
    print("References Cell not found.")
