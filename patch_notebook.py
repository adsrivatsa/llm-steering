import json

with open("llm_steering_3d.ipynb", "r") as f:
    nb = json.load(f)

# Find the first python cell or the one setting up env
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell["source"]
        # check if already inserted
        if any("miniconda_path" in line for line in source):
            break
        
        insert_code = [
            "import os\n",
            "import sys\n",
            "\n",
            "miniconda_path = f\"{os.environ['HOME']}/miniconda/bin\"\n",
            "os.environ[\"PATH\"] = f\"{miniconda_path}:\" + os.environ[\"PATH\"]\n",
            "\n",
            "print(f\"Conda PATH securely bound to: {miniconda_path}\")\n",
            "\n"
        ]
        
        # prepend to the first python cell
        nb["cells"][i]["source"] = insert_code + source
        break

with open("llm_steering_3d.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
