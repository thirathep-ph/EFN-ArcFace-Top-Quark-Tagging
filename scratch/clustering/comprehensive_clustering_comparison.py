"""
Comprehensive Clustering Comparison: 6+ Fair Methods for Hypersphere
=====================================================================
Each method independently determines optimal k.
Only includes methods compatible with hyperspherical geometry.

FAIR methods (cosine-compatible on sphere):
  1. Spectral Clustering (eigengap)
  2. K-means (elbow)
  3. Agglomerative (cosine, average linkage)
  4. DBSCAN (cosine)
  5. HDBSCAN (cosine) -- if installed
  6. DPGMM (Dirichlet Process)

UNFAIR methods (Euclidean -- included for comparison):
  7. GMM (BIC)
  8. Hierarchical (Ward)

CPU only -- uses main_arcface/embeddings.npz
"""
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_top import spectral_clustering_subclass

EMBEDDINGS_PATH = 'main_arcface/embeddings.npz'
N_SAMPLE = 30000

print("=" * 80)
print("COMPREHENSIVE CLUSTERING COMPARISON -- HYPERSPHERE-AWARE METHODS")
print("=" * 80)

# --- Load embeddings ---
data = np.load(EMBEDDINGS_PATH)
embeddings = data['embeddings']
labels = data['labels']
emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
print(f"\n[1] Loaded {len(embeddings):,} embeddings, dim={embeddings.shape[1]}")

# Subsample
rng = np.random.default_rng(42)
qcd_idx = np.where(labels == 0)[0]
top_idx = np.where(labels == 1)[0]
n_each = min(N_SAMPLE // 2, len(qcd_idx), len(top_idx))
sub_idx = np.concatenate([
    rng.choice(qcd_idx, n_each, replace=False),
    rng.choice(top_idx, n_each, replace=False)
])
emb_sub = emb_norm[sub_idx]
lab_sub = labels[sub_idx]
print(f"    Subsampled {len(sub_idx):,} ({n_each:,} per class)")

# Assign predictions via cosine similarity
top_center = emb_sub[lab_sub == 1].mean(axis=0)
top_center /= np.linalg.norm(top_center)
qcd_center = emb_sub[lab_sub == 0].mean(axis=0)
qcd_center /= np.linalg.norm(qcd_center)
cos_sim = np.dot(emb_sub, np.stack([qcd_center, top_center]).T)
preds = np.argmax(cos_sim, axis=1)
top_latent = emb_sub[preds == 1]
qcd_latent = emb_sub[preds == 0]
print(f"    Predicted Top: {len(top_latent):,}   Predicted QCD: {len(qcd_latent):,}")


def run_comparison(X, class_name):
    print(f"\n{'='*80}")
    print(f"  {class_name} Class ({len(X):,} samples)")
    print(f"{'='*80}")
    results = {}
    k_range = list(range(1, 11))

    # ====== FAIR METHODS ======

    # 1. Spectral Clustering (Eigengap)
    print("\n  [FAIR 1] Spectral Clustering (Eigengap) -- cosine affinity...")
    sc_labels, k_sc = spectral_clustering_subclass(X, max_k=10)
    results['SC(Eigengap)'] = {'k': k_sc, 'labels': sc_labels, 'fair': True}
    print(f"      -> k = {k_sc}")

    # 2. K-means (Elbow)
    print("  [FAIR 2] K-means (Elbow) -- L2=cosine on sphere...")
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X)
        inertias.append(km.inertia_)
    deltas = np.diff(inertias)
    if len(deltas) > 2:
        rate = np.diff(deltas)
        k_elbow = np.argmax(np.abs(rate)) + 2
    else:
        k_elbow = 2
    print(f"      Inertias: {[f'{i:.1f}' for i in inertias]}")
    print(f"      Elbow k = {k_elbow}")
    km_final = KMeans(n_clusters=k_elbow, random_state=42, n_init='auto')
    results['Kmeans(Elbow)'] = {'k': k_elbow, 'labels': km_final.fit_predict(X), 'fair': True}

    # 3. Agglomerative (cosine, average)
    print("  [FAIR 3] Agglomerative (cosine, average linkage)...")
    # Find best k by silhouette over k=2..10
    best_k, best_sil = 2, -1
    for k in range(2, 11):
        agg_k = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine')
        labs = agg_k.fit_predict(X)
        sil = silhouette_score(X, labs, metric='cosine')
        if sil > best_sil:
            best_k, best_sil = k, sil
    agg_labels = AgglomerativeClustering(n_clusters=best_k, linkage='average', metric='cosine').fit_predict(X)
    results['Agglo(Cos)'] = {'k': best_k, 'labels': agg_labels, 'fair': True}
    print(f"      Best silhouette k = {best_k} (sil={best_sil:.4f})")

    # 4. DBSCAN (cosine)
    print("  [FAIR 4] DBSCAN (cosine, auto eps via k-distance)...")
    nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X)
    k_dists = np.sort(nn.kneighbors(X)[0][:, -1])
    knee = np.argmax(np.diff(k_dists)) if len(k_dists) > 1 else len(k_dists)//2
    eps = max(k_dists[min(knee, len(k_dists)-1)], 0.01)
    db = DBSCAN(eps=eps, min_samples=10, metric='cosine')
    db_labels = db.fit_predict(X)
    n_clusters_db = len(set(db_labels) - {-1})
    n_noise = (db_labels == -1).sum()
    results['DBSCAN'] = {'k': n_clusters_db, 'labels': db_labels, 'fair': True}
    print(f"      eps={eps:.4f}, k={n_clusters_db}, noise={n_noise} ({100*n_noise/len(X):.1f}%)")

    # 5. HDBSCAN (if available)
    try:
        import hdbscan
        print("  [FAIR 5] HDBSCAN (cosine)...")
        hdb = hdbscan.HDBSCAN(min_cluster_size=max(20, len(X)//100), metric='cosine')
        hdb_labels = hdb.fit_predict(X)
        n_clusters_hdb = len(set(hdb_labels) - {-1})
        n_noise_hdb = (hdb_labels == -1).sum()
        results['HDBSCAN'] = {'k': n_clusters_hdb, 'labels': hdb_labels, 'fair': True}
        print(f"      k={n_clusters_hdb}, noise={n_noise_hdb} ({100*n_noise_hdb/len(X):.1f}%)")
    except ImportError:
        print("  [FAIR 5] HDBSCAN -- skipping (not installed)")
        results['HDBSCAN'] = {'k': 0, 'labels': None, 'fair': False}

    # 6. DPGMM (Dirichlet Process, spherical cov)
    print("  [FAIR 6] DPGMM (Dirichlet Process, spherical cov)...")
    dpgmm = BayesianGaussianMixture(
        n_components=10, weight_concentration_prior=1e-3,
        covariance_type='spherical', random_state=42, n_init=3
    )
    dpgmm_labels = dpgmm.fit_predict(X)
    active_k = np.sum(dpgmm.weights_ > 0.01)
    results['DPGMM'] = {'k': active_k, 'labels': dpgmm_labels, 'fair': True}
    print(f"      Active components (weight>0.01): {active_k}")
    print(f"      Weights: {[f'{w:.3f}' for w in dpgmm.weights_ if w > 0.01]}")

    # ====== UNFAIR METHODS ======

    # 7. GMM (BIC)
    print("\n  [UNFAIR 7] GMM (BIC) -- Gaussian assumption on sphere...")
    bics = [GaussianMixture(n_components=k, random_state=42, n_init=3).fit(X).bic(X) for k in k_range]
    k_bic = k_range[np.argmin(bics)]
    print(f"      BIC: k = {k_bic}")
    gmm_labels = GaussianMixture(n_components=k_bic, random_state=42, n_init=3).fit_predict(X)
    results['GMM(BIC)'] = {'k': k_bic, 'labels': gmm_labels, 'fair': False}

    # 8. Hierarchical (Ward)
    print("  [UNFAIR 8] Hierarchical (Ward) -- variance on sphere...")
    Z = linkage(X, method='ward')
    dists_z = Z[:, 2]
    if len(dists_z) > 2:
        k_ward = np.argmax(np.abs(np.diff(dists_z))) + 2
    else:
        k_ward = 2
    hier_labels = fcluster(Z, t=k_ward, criterion='maxclust') - 1
    results['Ward(Dendro)'] = {'k': k_ward, 'labels': hier_labels, 'fair': False}
    print(f"      Merge dists (first 10): {[f'{d:.3f}' for d in dists_z[:10]]}")
    print(f"      Ward k = {k_ward}")

    # ====== SUMMARY ======
    fair = [m for m, r in results.items() if r['fair'] and r['k'] > 1]
    unfair = [m for m, r in results.items() if not r['fair']]

    print(f"\n  {'='*60}")
    print(f"  k AGREEMENT -- FAIR METHODS:")
    print(f"  {'='*60}")
    for m in fair:
        tag = " *** DISAGREES" if results[m]['k'] != 2 else ""
        print(f"    {m:20s} -> k = {results[m]['k']}{tag}")

    print(f"\n  {'='*60}")
    print(f"  k AGREEMENT -- UNFAIR METHODS:")
    print(f"  {'='*60}")
    for m in unfair:
        print(f"    {m:20s} -> k = {results[m]['k']}")

    # ARI matrix
    # Only include methods with valid (non-None) labels in ARI matrix
    all_m = [m for m in fair + unfair if results[m]['labels'] is not None]
    print(f"\n  {'='*60}")
    print(f"  ARI MATRIX:")
    print(f"  {'='*60}")
    print(f"  {'':22s}", end="")
    for m2 in all_m:
        print(f"  {m2[:10]:10s}", end="")
    print()
    for m1 in all_m:
        print(f"  {m1[:20]:20s}", end="")
        for m2 in all_m:
            ari = adjusted_rand_score(results[m1]['labels'], results[m2]['labels'])
            print(f"  {ari:.4f}    ", end="")
        print()

    # Silhouette
    print(f"\n  {'='*60}")
    print(f"  SILHOUETTE SCORES (cosine metric):")
    print(f"  {'='*60}")
    for m in all_m:
        k = results[m]['k']
        if k > 1 and k < len(X):
            sil = silhouette_score(X, results[m]['labels'], metric='cosine')
            print(f"    {m:20s} (k={k:2d}): {sil:.4f}")
        else:
            print(f"    {m:20s} (k={k:2d}): N/A")

    return results


for cls_name, data in [("TOP", top_latent), ("QCD", qcd_latent)]:
    r = run_comparison(data, cls_name)
    if cls_name == "TOP":
        top_results = r
    else:
        qcd_results = r

print(f"\n{'='*80}")
print("FINAL k AGREEMENT SUMMARY")
print("="*80)
for cls, res in [("Top", top_results), ("QCD", qcd_results)]:
    print(f"\n  {cls}:")
    for m, r in res.items():
        print(f"    {m:20s} -> k={r['k']:2d}  [{'FAIR' if r['fair'] else 'UNFAIR'}]")
print(f"\n{'='*80}")
