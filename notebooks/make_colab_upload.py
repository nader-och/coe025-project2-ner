"""Build colab_upload.zip: the code + the three small trained classical models.

The Colab notebook needs the code (data/, src/, tests/) and the models of
methods 1-3 (for the timing on the same machine). Raw data is downloaded again
in Colab from Hugging Face.
Run from the project root:  python notebooks/make_colab_upload.py
"""
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    files = (glob.glob("data/*.py") + glob.glob("src/*.py") + glob.glob("tests/*.py")
             + ["requirements.txt", "models/baseline.json", "models/hmm.json", "models/crf.pt"])
    out = os.path.join(ROOT, "colab_upload.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), f.replace("\\", "/"))
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
