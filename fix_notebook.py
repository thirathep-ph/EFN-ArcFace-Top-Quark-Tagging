import json

nb_path = 'f:/Jet Tagging/Top Tagging/TopTagging_ArcFace_Colab.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Install Dependencies
nb['cells'][1]['source'] = [
    "# 1. Install Dependencies\n",
    "%pip install -q awkward fastjet h5py hdf5plugin vector scikit-learn matplotlib seaborn"
]

# Cell 3: Imports
nb['cells'][3]['source'] = [
    "import json\n",
    "import time\n",
    "import numpy as np\n",
    "import awkward as ak\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "import hdf5plugin\n",
    "import h5py\n",
    "import vector\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score\n",
    "import fastjet\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")"
]

# Cell 4: load_awkward
nb['cells'][5]['source'] = [
    "def load_awkward(h5_path, max_events=None, trim_padding=True):\n",
    "    print(f\"Reading {h5_path}...\")\n",
    "    with h5py.File(h5_path, 'r') as h5_py:\n",
    "        # Top Tagging dataset has a nested 'table/table' structure\n",
    "        if 'table' in h5_py and 'table' in h5_py['table']:\n",
    "            table = h5_py['table']['table']\n",
    "            n = table.shape[0] if max_events is None else min(max_events, table.shape[0])\n",
    "            \n",
    "            # constituents: values_block_0 [0:800] = 200 x (E, PX, PY, PZ) interleaved\n",
    "            v0 = table['values_block_0'][:n]\n",
    "            v1 = table['values_block_1'][:n]\n",
    "            \n",
    "            labels = v1[:, 1].astype(np.int32)\n",
    "            p4_data = np.stack([\n",
    "                v0[:, 0:800:4], # E\n",
    "                v0[:, 1:800:4], # px\n",
    "                v0[:, 2:800:4], # py\n",
    "                v0[:, 3:800:4]  # pz\n",
    "            ], axis=2).astype(np.float32)\n",
    "        else:\n",
    "            # Fallback for flat datasets\n",
    "            ds = h5_py['table']\n",
    "            n = ds.shape[0] if max_events is None else min(max_events, ds.shape[0])\n",
    "            data = ds[:n]\n",
    "            if data.ndim == 2:\n",
    "                labels = data[:, 800].astype(np.int32) if data.shape[1] > 800 else np.zeros(n)\n",
    "                p4_data = data[:, :800].reshape(-1, 200, 4).astype(np.float32)\n",
    "            else:\n",
    "                p4_data = data.astype(np.float32)\n",
    "                labels = np.zeros(n)\n",
    "        \n",
    "        weights = np.ones(len(labels), dtype=np.float32)\n",
    "    \n",
    "    if trim_padding:\n",
    "        mask = p4_data[:, :, 0] > 0\n",
    "        counts = np.sum(mask, axis=1)\n",
    "        p4_flat = p4_data[mask]\n",
    "        events = ak.unflatten(ak.zip({\n",
    "            \"E\": p4_flat[:, 0], \"px\": p4_flat[:, 1], \"py\": p4_flat[:, 2], \"pz\": p4_flat[:, 3]\n",
    "        }, with_name=\"Momentum4D\"), counts)\n",
    "    else:\n",
    "        events = ak.zip({\n",
    "            \"E\": p4_data[:, :, 0], \"px\": p4_data[:, :, 1], \"py\": p4_data[:, :, 2], \"pz\": p4_data[:, :, 3]\n",
    "        }, with_name=\"Momentum4D\")\n",
    "        \n",
    "    return events, labels, weights\n",
    "\n",
    "def compute_features(events):\n",
    "    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)\n",
    "    cluster = fastjet.ClusterSequence(events, jet_def)\n",
    "    jets = cluster.inclusive_jets(min_pt=200)\n",
    "    jets = jets[ak.argsort(jets.pt, axis=-1, ascending=False)[:, :1]]\n",
    "    mass = ak.to_numpy(ak.flatten(jets.mass))\n",
    "    pt = ak.to_numpy(ak.flatten(jets.pt))\n",
    "    return np.stack([mass, pt], axis=1)"
]

# Cell 6: JetDataset & run_train
nb['cells'][11]['source'] = [
    "class JetDataset(Dataset):\n",
    "    def __init__(self, h5, max_events=None):\n",
    "        # Use trim_padding=False to get fixed-size arrays directly\n",
    "        self.ev, self.lab, _ = load_awkward(h5, max_events, trim_padding=False)\n",
    "        self.n = len(self.ev)\n",
    "        # Convert awkward array to numpy efficiently without looping\n",
    "        self.c = np.stack([\n",
    "            ak.to_numpy(self.ev.E),\n",
    "            ak.to_numpy(self.ev.px),\n",
    "            ak.to_numpy(self.ev.py),\n",
    "            ak.to_numpy(self.ev.pz)\n",
    "        ], axis=2).astype(np.float32)\n",
    "        self.m = self.c[:, :, 0] > 0\n",
    "    def __len__(self): return self.n\n",
    "    def __getitem__(self, idx): return self.c[idx], self.lab[idx], self.m[idx]\n",
    "\n",
    "def run_train():\n",
    "    train_ds = JetDataset(train_h5, max_events=None) # Full Training\n",
    "    val_ds = JetDataset(val_h5, max_events=100000)\n",
    "    loader = DataLoader(train_ds, batch_size=1024, shuffle=True)\n",
    "    v_loader = DataLoader(val_ds, batch_size=1024)\n",
    "    model = TopTaggingModel().to(device)\n",
    "    opt = optim.Adam(model.parameters(), lr=1e-3)\n",
    "    for ep in range(30):\n",
    "        model.train()\n",
    "        for x, y, m in loader:\n",
    "            x, y, m = x.to(device), y.to(device), m.to(device)\n",
    "            opt.zero_grad(); l, _ = model(x, y, m); loss = F.cross_entropy(l, y); loss.backward(); opt.step()\n",
    "        model.eval(); acc = 0\n",
    "        with torch.no_grad():\n",
    "            for x, y, m in v_loader:\n",
    "                l, _ = model(x.to(device), m=m.to(device)); acc += (l.argmax(1) == y.to(device)).float().mean().item()\n",
    "        print(f\"Epoch {ep+1}: Val Acc={acc/len(v_loader):.4f}\")\n",
    "        torch.save(model.state_dict(), 'best_model.pt')\n",
    "\n",
    "run_train()"
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
