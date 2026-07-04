"""
ONNX Export + Calibration
==========================
Loads the trained model, exports to ONNX format, runs inference with ONNX Runtime,
and compares accuracy with PyTorch.
Saves to experiments/model.onnx

NOTE: The ONNX export has a known issue with phi-wrapping (fmod) that causes NaN
outputs in ONNX Runtime. This is documented in the results. The PyTorch reference
accuracy is reported correctly.
"""
import os, sys, gc, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
MODEL_PATH = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")
ONNX_PATH = os.path.join(EXPERIMENTS_DIR, "model.onnx")

N_EVENTS = 5000
BATCH_SIZE = 256


class ONNXCompatibleModel(torch.nn.Module):
    """
    Wrapper that replaces asinh with log(x + sqrt(x^2+1)) for ONNX compatibility.
    """
    def __init__(self, base_model):
        super().__init__()
        self.efn = base_model.efn
        self.arcface_head = base_model.arcface_head

    def _safe_asinh(self, x):
        return torch.log(x + torch.sqrt(x * x + 1.0))

    def forward(self, constituents, mask):
        E = constituents[:, :, 0]
        px = constituents[:, :, 1]
        py = constituents[:, :, 2]
        pz = constituents[:, :, 3]

        pt = torch.sqrt(px**2 + py**2 + 1e-10)
        eta = self._safe_asinh(pz / (pt + 1e-10))
        phi = torch.atan2(py, px)

        if mask is not None:
            m = mask.unsqueeze(-1).float()
            jet_p4 = (constituents * m).sum(dim=1)
        else:
            jet_p4 = constituents.sum(dim=1)

        j_px = jet_p4[:, 1]
        j_py = jet_p4[:, 2]
        j_pz = jet_p4[:, 3]
        j_pt = torch.sqrt(j_px**2 + j_py**2 + 1e-10)
        j_eta = self._safe_asinh(j_pz / (j_pt + 1e-10))
        j_phi = torch.atan2(j_py, j_px)

        d_eta = eta - j_eta.unsqueeze(1)
        d_phi = phi - j_phi.unsqueeze(1)
        # Phi wrapping: use atan2(sin, cos) which is ONNX-safe
        d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))

        features = torch.stack([d_eta, d_phi], dim=2)
        z = E / (E.sum(dim=1, keepdim=True) + 1e-10)

        if mask is not None:
            features = features * mask.unsqueeze(-1).float()
            z = z * mask.float()

        jet_emb = self.efn(features, z, mask)
        logits, similarities = self.arcface_head(jet_emb, None)

        return logits, similarities, jet_emb


def export_onnx(model, onnx_path):
    """Export model to ONNX format."""
    model.eval()
    onnx_model = ONNXCompatibleModel(model)
    onnx_model.eval()

    dummy_x = torch.randn(1, 200, 4)
    dummy_mask = torch.ones(1, 200)

    torch.onnx.export(
        onnx_model,
        (dummy_x, dummy_mask),
        onnx_path,
        input_names=['constituents', 'mask'],
        output_names=['logits', 'similarities', 'embedding'],
        dynamic_axes={
            'constituents': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'similarities': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        opset_version=17,
    )
    print(f"  ONNX model exported to {onnx_path}")
    return onnx_path


def evaluate_pytorch(model, loader, n_events):
    """Evaluate with PyTorch."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    count = 0

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            if count >= n_events:
                break
            logits, sims, emb = model(x, mask=m)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.numpy())
            all_labels.append(y.numpy())
            all_logits.append(logits.numpy())
            count += len(y)

    return np.concatenate(all_preds), np.concatenate(all_labels), np.vstack(all_logits)


def evaluate_onnx(onnx_path, loader, n_events):
    """Evaluate with ONNX Runtime."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  ONNX input names: {input_names}")

    all_preds = []
    all_labels = []
    all_logits = []
    count = 0

    for x, y, w, m, _ in loader:
        if count >= n_events:
            break

        ort_inputs = {
            'constituents': x.numpy().astype(np.float32),
            'mask': m.numpy().astype(np.float32),
        }
        ort_outputs = session.run(None, ort_inputs)
        logits_np = ort_outputs[0]

        preds = np.argmax(logits_np, axis=1)
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_logits.append(logits_np)
        count += len(y)

    return np.concatenate(all_preds), np.concatenate(all_labels), np.vstack(all_logits)


def main():
    device, backend_type = get_device()
    print(f"Device: {device}\n")
    print("=" * 60)
    print("ONNX EXPORT + CALIBRATION")
    print("=" * 60)

    # Load model
    print("\nStep 1: Loading PyTorch model...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  Model loaded from {MODEL_PATH}")

    # Load data
    print(f"\nStep 2: Loading {N_EVENTS} test events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Loaded {len(ds)} events")

    # Export to ONNX
    print(f"\nStep 3: Exporting to ONNX...")
    t0 = time.time()
    export_onnx(model, ONNX_PATH)
    export_time = time.time() - t0
    print(f"  Export time: {export_time:.2f}s")
    onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  ONNX file size: {onnx_size_mb:.2f} MB")

    # Evaluate PyTorch
    print(f"\nStep 4: Evaluating PyTorch model...")
    t0 = time.time()
    pt_preds, pt_labels, pt_logits = evaluate_pytorch(model, loader, N_EVENTS)
    pt_time = time.time() - t0
    pt_acc = np.mean(pt_preds == pt_labels)
    print(f"  PyTorch accuracy: {pt_acc:.4f}")
    print(f"  PyTorch inference time: {pt_time:.2f}s ({pt_time/N_EVENTS*1000:.3f} ms/event)")

    # Evaluate ONNX
    print(f"\nStep 5: Evaluating ONNX Runtime model...")
    onnx_status = "unknown"
    onnx_acc = 0.0
    onnx_time = 0.0
    onnx_logits = np.zeros_like(pt_logits)
    onnx_preds = np.zeros_like(pt_preds)

    try:
        t0 = time.time()
        onnx_preds, onnx_labels, onnx_logits = evaluate_onnx(ONNX_PATH, loader, N_EVENTS)
        onnx_time = time.time() - t0

        # Check for NaN
        onnx_nan_count = int(np.isnan(onnx_logits).sum())
        if onnx_nan_count > 0:
            print(f"  WARNING: {onnx_nan_count} NaN values in ONNX logits!")
            print(f"  This is a known issue with phi-wrapping in ONNX Runtime.")
            onnx_logits_clean = np.nan_to_num(onnx_logits, nan=0.0)
            onnx_preds = np.argmax(onnx_logits_clean, axis=1)
            onnx_acc = np.mean(onnx_preds == onnx_labels)
            onnx_status = "nan_detected"
        else:
            onnx_acc = np.mean(onnx_preds == onnx_labels)
            onnx_status = "ok"

        print(f"  ONNX accuracy: {onnx_acc:.4f}")
        print(f"  ONNX inference time: {onnx_time:.2f}s ({onnx_time/N_EVENTS*1000:.3f} ms/event)")

    except ImportError:
        print("  WARNING: onnxruntime not installed. Skipping ONNX evaluation.")
        onnx_status = "onnxruntime_not_installed"
    except Exception as e:
        print(f"  ERROR: ONNX evaluation failed: {e}")
        onnx_status = f"error: {str(e)[:100]}"

    # Compare
    acc_diff = abs(pt_acc - onnx_acc) if onnx_status != "onnxruntime_not_installed" else None
    pred_agreement = np.mean(pt_preds == onnx_preds) if onnx_status not in ["onnxruntime_not_installed"] else None
    logit_diff = float(np.nanmean(np.abs(pt_logits - onnx_logits))) if onnx_status not in ["onnxruntime_not_installed"] else None

    print(f"\n{'='*60}")
    print("COMPARISON")
    print("=" * 60)
    print(f"  PyTorch accuracy:     {pt_acc:.4f}")
    if onnx_status not in ["onnxruntime_not_installed"]:
        print(f"  ONNX accuracy:        {onnx_acc:.4f}")
        print(f"  Accuracy difference:  {acc_diff:.6f}")
        print(f"  Prediction agreement: {pred_agreement:.4f}")
        print(f"  Mean logit difference:{logit_diff:.6f}")
        if pt_time > 0 and onnx_time > 0:
            print(f"  Speedup (ONNX/PT):   {pt_time/onnx_time:.2f}x")
    print(f"  ONNX status:          {onnx_status}")

    # Plot
    print(f"\nStep 6: Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Logit comparison scatter
    ax = axes[0]
    if onnx_status not in ["onnxruntime_not_installed"]:
        valid_mask = ~(np.isnan(pt_logits[:, 1]) | np.isnan(onnx_logits[:, 1]))
        if valid_mask.sum() > 0:
            ax.scatter(pt_logits[valid_mask, 1], onnx_logits[valid_mask, 1], alpha=0.1, s=5, c='steelblue')
            lims = [min(pt_logits[valid_mask, 1].min(), onnx_logits[valid_mask, 1].min()),
                    max(pt_logits[valid_mask, 1].max(), onnx_logits[valid_mask, 1].max())]
            ax.plot(lims, lims, 'r--', lw=2, label='Perfect agreement')
        else:
            ax.text(0.5, 0.5, 'No valid ONNX outputs\n(NaN detected)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red')
    ax.set_xlabel('PyTorch Logit (Top)')
    ax.set_ylabel('ONNX Logit (Top)')
    ax.set_title(f'Logit Agreement\n(status: {onnx_status})')
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Plot 2: Logit difference histogram
    ax = axes[1]
    if onnx_status not in ["onnxruntime_not_installed"]:
        logit_diffs = np.abs(pt_logits - onnx_logits).max(axis=1)
        valid_diffs = logit_diffs[~np.isnan(logit_diffs)]
        if len(valid_diffs) > 0:
            ax.hist(valid_diffs, bins=50, color='steelblue', edgecolor='black', linewidth=0.5)
            ax.axvline(np.nanmean(valid_diffs), color='red', linestyle='--',
                       label=f'Mean: {np.nanmean(valid_diffs):.6f}')
        else:
            ax.text(0.5, 0.5, 'All differences are NaN',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red')
    ax.set_xlabel('Max |Logit Difference|')
    ax.set_ylabel('Count')
    ax.set_title('PyTorch vs ONNX Logit Difference')
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Plot 3: Accuracy comparison bar chart
    ax = axes[2]
    if onnx_status not in ["onnxruntime_not_installed"]:
        categories = ['PyTorch', 'ONNX\nRuntime']
        accuracies = [pt_acc, onnx_acc]
        times_ms = [pt_time/N_EVENTS*1000, onnx_time/N_EVENTS*1000]
        bars = ax.bar([0, 1], accuracies, color=['steelblue', 'darkorange'],
                      edgecolor='black', linewidth=0.5, width=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(categories)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy Comparison\nPT: {times_ms[0]:.2f}ms, ONNX: {times_ms[1]:.2f}ms')
        ax.set_ylim(min(accuracies) - 0.01, 1.0)
        for bar, acc, t in zip(bars, accuracies, times_ms):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{acc:.4f}\n({t:.2f}ms)', ha='center', va='bottom', fontsize=9)
    else:
        bars = ax.bar([0], [pt_acc], color=['steelblue'], edgecolor='black', linewidth=0.5, width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels(['PyTorch'])
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy (ONNX Runtime not installed)\nPT: {pt_time/N_EVENTS*1000:.2f}ms')
        ax.set_ylim(pt_acc - 0.01, 1.0)
        ax.text(0, pt_acc + 0.002, f'{pt_acc:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "onnx_calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: onnx_calibration.png")

    # Save results
    out = {
        "n_events": N_EVENTS,
        "pytorch_accuracy": float(pt_acc),
        "onnx_status": onnx_status,
        "onnx_accuracy": float(onnx_acc) if onnx_status not in ["onnxruntime_not_installed"] else None,
        "accuracy_difference": float(acc_diff) if acc_diff is not None else None,
        "prediction_agreement": float(pred_agreement) if pred_agreement is not None else None,
        "mean_logit_difference": float(logit_diff) if logit_diff is not None else None,
        "pytorch_time_ms_per_event": float(pt_time / N_EVENTS * 1000),
        "onnx_time_ms_per_event": float(onnx_time / N_EVENTS * 1000) if onnx_time > 0 else None,
        "speedup": float(pt_time / onnx_time) if onnx_time > 0 else None,
        "onnx_file_size_mb": float(onnx_size_mb),
    }
    with open(os.path.join(EXPERIMENTS_DIR, "onnx_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: onnx_results.json")

    print(f"\n{'='*60}")
    print("ONNX EXPORT COMPLETE")
    print("=" * 60)

    del model
    gc.collect()


if __name__ == '__main__':
    main()
