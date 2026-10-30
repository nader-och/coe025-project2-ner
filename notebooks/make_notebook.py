"""Write notebooks/colab_bert.ipynb (the Colab notebook for methods 4 and 5).

Week 1: Google Colab gives a free GPU. BERT training needs a GPU.
Run from the project root:  python notebooks/make_notebook.py
"""
import json
import os

CELLS = [
    ("markdown", """# COE025 Project 2 - BERT methods on Google Colab

Week 1 recommends Google Colab for a free GPU. This notebook trains:
- **Method 4: BERT + softmax** (`src/bert_softmax.py`, Week 6 Slides 82-86)
- **Method 5: BERT + CRF** (`src/bert_crf.py`, Week 6 Slides 79-82)

and measures the prediction time of all five methods on this machine (`src/timing.py`).

**Before you start:** menu *Runtime -> Change runtime type -> T4 GPU -> Save*.
Then run the cells one by one from top to bottom (Shift+Enter)."""),
    ("code", "# Step 1: check that a GPU is available (you should see 'Tesla T4')\n!nvidia-smi"),
    ("code", """# Step 2: upload colab_upload.zip (it is in the project folder on your computer)
from google.colab import files
import zipfile
uploaded = files.upload()
zip_name = list(uploaded)[0]
zipfile.ZipFile(zip_name).extractall('/content/project')
%cd /content/project
!ls"""),
    ("code", "# Step 3: install the libraries (Colab already has PyTorch)\n"
             "!pip install -q transformers huggingface_hub pytest"),
    ("code", "# Step 4: run the unit tests (they must all pass)\n!python -m pytest -q tests"),
    ("code", "# Step 5: Method 4, BERT + softmax (about 5-10 minutes on a T4)\n"
             "!python -m src.bert_softmax"),
    ("code", "# Step 6: Method 5, BERT + CRF (about 5-10 minutes on a T4)\n"
             "!python -m src.bert_crf"),
    ("code", "# Step 7: prediction time of all 5 methods on this same machine\n"
             "!python -m src.timing --tag colab"),
    ("code", """# Step 8: download the results (a file named results_colab.zip)
import shutil
shutil.make_archive('/content/results_colab', 'zip', '/content/project', 'results')
files.download('/content/results_colab.zip')"""),
    ("markdown", """**Done.** Put `results_colab.zip` in the project folder on your computer
and unzip it there (it fills `results/`). Then run the final evaluation steps in the README.

The trained BERT weights stay in Colab (`/content/project/models`). They are large
and are not needed for the next steps."""),
]


def main():
    nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_bert.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
