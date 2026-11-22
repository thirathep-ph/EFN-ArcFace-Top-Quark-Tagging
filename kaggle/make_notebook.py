import json

nb = {
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
  },
  "cells": []
}

def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})

md("# ArcEFN - Top-Quark Tagging: train on Kaggle\n\n"
   "Reproduces EFN+ArcFace training for an arbitrary seed on Kaggle's free GPU.\n"
   "Set SEED in the config cell. Outputs (checkpoint + history) are zipped for download.")

md("## 0. Setup")

code("!nvidia-smi || true\n!python --version")

code("# Install deps not pre-installed on Kaggle\n!pip install -q --upgrade pip\n!pip install -q awkward vector h5py hdf5plugin tqdm")

code("""import os, subprocess, sys
REPO = "/kaggle/working/ArcEFN"
URL = "https://github.com/thirathep-ph/EFN-ArcFace-Top-Quark-Tagging.git"
if os.path.isdir(REPO):
    subprocess.run(["git", "-C", REPO, "fetch", "origin"], check=False)
    subprocess.run(["git", "-C", REPO, "reset", "--hard", "origin/main"], check=False)
else:
    subprocess.run(["git", "clone", URL, REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
print("cwd:", os.getcwd())
print("git:", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())""")

md("## 1. Config")

code("""SEED = 67          # change to 404 for the second run
EPOCHS = 50
MAX_EVENTS = None   # set e.g. 50000 for a quick sanity run
# Keep False to match the canonical FP32 runs (AMD/DirectML has no AMP).
# Set True only if you need speed and accept tiny numerical differences.
AMP = False
BATCH_SIZE = 128
ACCUM = 8
CHUNK_SIZE = 100000
CKPT_DIR = "experiments/arcface_seed_{}".format(SEED)
OUT_DIR = "experiments/arcface_seed_{}".format(SEED)

import os
os.environ["ARCEFN_MODEL_DIR"] = CKPT_DIR
print("seed", SEED, "epochs", EPOCHS)""")

md("## 2. Data (Top Tagging, 3.4 GB from Zenodo)")

code("""import os, glob
# Prefer an attached Kaggle Dataset (upload once) over re-downloading from Zenodo.
attached = []
for root, _, files in os.walk("/kaggle/input"):
    for fn in files:
        if fn.endswith(".h5"):
            attached.append(os.path.join(root, fn))
if attached:
    DATA_DIR = os.path.dirname(attached[0])
    os.environ["ARCEFN_DATA_DIR"] = DATA_DIR
    print("Using attached dataset:", DATA_DIR, attached)
else:
    DATA_DIR = "data/top_tagging"
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".h5")]
    total = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in existing) if existing else 0
    print("existing h5 bytes:", total)
    if total < 1_700_000_000:
        get_ipython().system("python scripts/download_data.py --dest " + DATA_DIR)
    else:
        print("dataset already present, skipping download")""")

code("# verify\nimport os\nprint(sorted(os.listdir('data/top_tagging')))")

md("## 3. Train")

code("""import os
cmd = [
    "python", "scripts/train_arcface.py",
    "--seed", str(SEED),
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--accumulation_steps", str(ACCUM),
    "--chunk_size", str(CHUNK_SIZE),
    "--checkpoint_dir", CKPT_DIR,
    "--output_dir", OUT_DIR,
]
if MAX_EVENTS:
    cmd += ["--max_events", str(MAX_EVENTS)]
if AMP:
    cmd += ["--amp"]
print(" ".join(cmd))
get_ipython().system(" ".join(cmd))""")

md("## 4. Inspect + package outputs")

code("""import json, os, glob
print("files:")
for f in sorted(glob.glob(CKPT_DIR + "/**/*", recursive=True)):
    print(" ", f, os.path.getsize(f) if os.path.isfile(f) else "")
hp = os.path.join(OUT_DIR, "history.json")
if os.path.exists(hp):
    h = json.load(open(hp))
    print("final val_acc:", h["val_acc"][-1] if h.get("val_acc") else None)
    print("best val_acc:", max(h["val_acc"]) if h.get("val_acc") else None)""")

code("""# zip for download from the Output tab
import shutil, os
zip_path = "/kaggle/working/arcefn_seed_{}.zip".format(SEED)
shutil.make_archive(zip_path.replace(".zip",""), "zip", CKPT_DIR)
print("zip:", zip_path, os.path.getsize(zip_path)/1e6, "MB")""")

with open("kaggle/train_arcefn.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("written")