"""scbench: helper functions for the scRNA-seq embedding benchmark.

Do foundation models beat classical pipelines on small scRNA-seq datasets?
Compares PCA, PCA + Harmony, scVI and zero-shot Geneformer embeddings with scIB-style metrics.

Everything the notebook needs is here (data loading, preprocessing, embeddings helpers,
metrics, plotting), so the notebook itself stays a readable narrative.
"""
import os, random, gc, hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
                             silhouette_samples, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SEED = 0
SEEDS = (0, 1, 2)                     # replicate seeds: Leiden, CV folds, and scVI training
CFG = dict(
    n_hvg=2000, n_pcs=50, n_neighbors=15, knn_k=15, cv_folds=5,
    resolutions=[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8],   # pancreas Leiden grid
    pbmc_resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8],              # grid used for the PBMC3k sanity check
    out_dir="/kaggle/working/results",
)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_pbmc3k():
    """Raw counts from pbmc3k + cell-type labels transferred from pbmc3k_processed (by barcode)."""
    raw = sc.datasets.pbmc3k()
    lab = sc.datasets.pbmc3k_processed()
    keep = raw.obs_names.intersection(lab.obs_names)
    raw = raw[keep].copy()
    raw.obs["cell_type"] = lab.obs.loc[keep, "louvain"].astype(str).values
    raw.var_names_make_unique()
    raw.layers["counts"] = raw.X.copy()
    raw.obs["batch"] = "pbmc3k"
    return raw


def figshare_download(article_id, fname, dest):
    """Download one file from a figshare article through the public API and verify its MD5."""
    import requests
    art = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=60).json()
    print("article:", art["title"], "| version:", art.get("version"), "| doi:", art.get("doi"))
    f = {x["name"]: x for x in art["files"]}[fname]
    if not os.path.exists(dest):
        with requests.get(f["download_url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
    md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
    print("md5 match:", md5 == f["computed_md5"], "|", md5)
    assert md5 == f["computed_md5"], "Checksum mismatch: delete the file and re-run."


def load_pancreas(path, batch_col, label_col, drop_batches, round_batches, max_cells=None, seed=SEED):
    """scIB pancreas: drop batches whose `counts` layer is not count-like, round near-integer batches,
    verify every remaining value is a non-negative integer, return a clean sparse-count AnnData."""
    a = sc.read_h5ad(path)
    b_all = a.obs[batch_col].astype(str)
    y_all = a.obs[label_col].astype(str)
    keep = ~b_all.isin(drop_batches).values
    counts = np.asarray(a.layers["counts"])[keep]
    obs = pd.DataFrame({"batch": b_all[keep].values, "cell_type": y_all[keep].values}, index=a.obs_names[keep])
    var_names = a.var_names.astype(str)
    lost = sorted(set(y_all) - set(obs["cell_type"]))
    n_removed = int((~keep).sum())
    del a
    gc.collect()
    print(f"dropped batches {drop_batches}: {n_removed} cells removed -> {obs.shape[0]} remain")
    print("cell types lost entirely by the exclusion:", lost or "none")

    rb = obs["batch"].isin(round_batches).values
    if rb.any():
        before = counts[rb]
        after = np.round(before)
        print(f"rounded {int(rb.sum())} cells in {round_batches}; max |change| = {np.abs(after - before).max():.3f}")
        counts[rb] = after
        del before, after

    assert counts.min() >= 0, "negative values in counts"
    for i in range(0, counts.shape[0], 2000):
        ch = counts[i:i + 2000]
        bad = ~np.isclose(ch, np.round(ch))
        if bad.any():
            raise AssertionError("non-integer values remain in batches: "
                                 f"{obs['batch'].iloc[i:i + 2000][bad.any(1)].unique().tolist()}")

    X = sp.csr_matrix(counts)
    del counts
    gc.collect()
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    out.var_names_make_unique()
    out.layers["counts"] = out.X.copy()

    if max_cells and out.n_obs > max_cells:
        frac, idx = max_cells / out.n_obs, []
        for _, d in out.obs.groupby(["batch", "cell_type"], observed=True):
            idx += list(d.sample(n=max(1, int(round(len(d) * frac))), random_state=seed).index)
        out = out[idx].copy()
        print(f"subsampled to {out.n_obs} cells")
    return out


# --------------------------------------------------------------------------------------
# Preprocessing and classical embeddings
# --------------------------------------------------------------------------------------
def preprocess(adata, n_hvg=CFG["n_hvg"], n_pcs=CFG["n_pcs"], seed=SEED, hvg_batch_key=None, max_pct_mt=5):
    """QC -> HVG (seurat_v3 on raw counts, optionally batch-aware) -> normalize/log1p -> scale -> PCA.
    Returns the HVG-subset AnnData; the full-gene log-normalised matrix is kept in `.raw`."""
    adata = adata.copy()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    n0 = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    if max_pct_mt is not None:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()
    print(f"cells: {n0} -> {adata.n_obs} | genes: {adata.n_vars}")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3",
                                layer="counts", batch_key=hvg_batch_key)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return adata


def run_harmony(adata, batch_key, basis="X_pca", out_key="X_pca_harmony", seed=SEED):
    import harmonypy
    try:
        ho = harmonypy.run_harmony(adata.obsm[basis], adata.obs, batch_key, random_state=seed)
    except TypeError:
        ho = harmonypy.run_harmony(adata.obsm[basis], adata.obs, batch_key)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != adata.n_obs:                     # harmonypy has returned both orientations across versions
        Z = Z.T
    assert Z.shape == adata.obsm[basis].shape, Z.shape
    adata.obsm[out_key] = Z


def graph_only(adata, rep, name, n_neighbors=CFG["n_neighbors"], seed=SEED):
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors, key_added=name, random_state=seed)


def graph_umap(adata, rep, name, n_neighbors=CFG["n_neighbors"], seed=SEED):
    graph_only(adata, rep, name, n_neighbors, seed)
    sc.tl.umap(adata, neighbors_key=name, random_state=seed)
    adata.obsm[f"X_umap_{name}"] = adata.obsm["X_umap"].copy()


def ensure_graph(adata, rep, name, umap=False):
    """Build the kNN graph (and optionally the UMAP) only if it does not exist yet."""
    if f"{name}_connectivities" not in adata.obsp:
        (graph_umap if umap else graph_only)(adata, rep, name)
    elif umap and f"X_umap_{name}" not in adata.obsm:
        graph_umap(adata, rep, name)


# --------------------------------------------------------------------------------------
# Metrics (scIB-style)
# --------------------------------------------------------------------------------------
def batch_asw(X, batches, labels):
    """scIB batch ASW: per cell type, 1 - |silhouette of batch labels|, averaged. 1 = well mixed."""
    scores = []
    for lab in np.unique(labels):
        m = labels == lab
        nb = len(np.unique(batches[m]))
        if nb < 2 or m.sum() <= nb:
            continue                                   # cell type lives in one batch: not scoreable
        scores.append((1 - np.abs(silhouette_samples(X[m], batches[m]))).mean())
    return float(np.mean(scores))


def graph_connectivity(adata, name, label_key="cell_type"):
    """scIB graph connectivity: per cell type, fraction in the largest connected component of the kNN graph."""
    A, y, scores = adata.obsp[f"{name}_connectivities"], adata.obs[label_key].values, []
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        _, comp = connected_components(A[idx][:, idx], directed=False)
        scores.append(np.bincount(comp).max() / len(idx))
    return float(np.mean(scores))


def cluster_scores(adata, name, label_key="cell_type", seed=SEED, resolutions=None):
    """Leiden resolution sweep (best NMI against the labels, scIB convention) for one seed."""
    resolutions = resolutions or CFG["resolutions"]
    best = None
    for r in resolutions:
        sc.tl.leiden(adata, resolution=r, neighbors_key=name, key_added="_tmp_leiden",
                     flavor="igraph", n_iterations=2, directed=False, random_state=seed)
        cl, y = adata.obs["_tmp_leiden"], adata.obs[label_key]
        nmi = normalized_mutual_info_score(y, cl)
        if best is None or nmi > best["NMI"]:
            best = {"NMI": nmi, "ARI": adjusted_rand_score(y, cl), "best_res": r, "n_clusters": int(cl.nunique())}
    del adata.obs["_tmp_leiden"]
    return best


def knn_macro_f1(X, y, k=CFG["knn_k"], folds=CFG["cv_folds"], seed=SEED):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=k), X, y, cv=cv)
    return f1_score(y, pred, average="macro")


def _lr():
    """Class-balanced logistic regression on standardised features (so PCA, scVI and Geneformer
    embeddings of different scale are treated alike)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def annot_cv(X, y, seed, folds=CFG["cv_folds"]):
    """Annotation with stratified CV: kNN and balanced logistic regression macro-F1."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(_lr(), X, y, cv=cv, n_jobs=-1)
    return {"kNN_macroF1": knn_macro_f1(X, y, seed=seed), "LR_macroF1": f1_score(y, pred, average="macro")}


def annot_lobo(X, y, b, labels=None):
    """Leave-one-batch-out: train on all other batches, predict the held-out one (pooled macro-F1)."""
    pred = cross_val_predict(_lr(), X, y, cv=LeaveOneGroupOut(), groups=b, n_jobs=-1)
    return f1_score(y, pred, labels=labels, average="macro")


def _geometry_metrics(adata, rep, name, label_key, batch_key):
    X, y, b = adata.obsm[rep], adata.obs[label_key].values, adata.obs[batch_key].values
    return X, y, b, {"cell_type_ASW": (silhouette_score(X, y) + 1) / 2,
                     "batch_ASW": batch_asw(X, b, y),
                     "graph_conn": graph_connectivity(adata, name, label_key)}


def evaluate_all(adata, rep, name, label_key="cell_type", batch_key="batch", seeds=SEEDS):
    """Full evaluation, one row per seed. The seed varies Leiden and the CV folds;
    ASW, graph connectivity and leave-one-batch-out are deterministic given the embedding."""
    X, y, b, fixed = _geometry_metrics(adata, rep, name, label_key, batch_key)
    fixed["LR_LOBO_macroF1"] = annot_lobo(X, y, b)
    rows = [{"method": name, "seed": s, **cluster_scores(adata, name, label_key, s),
             **annot_cv(X, y, s), **fixed} for s in seeds]
    return pd.DataFrame(rows)


def evaluate_light(adata, rep, name, label_key="cell_type", batch_key="batch", seeds=SEEDS):
    """Clustering + geometry + kNN only (no logistic regression); used for sensitivity runs."""
    X, y, b, fixed = _geometry_metrics(adata, rep, name, label_key, batch_key)
    return pd.DataFrame([{"method": name, "seed": s, **cluster_scores(adata, name, label_key, s),
                          "kNN_macroF1": knn_macro_f1(X, y, seed=s), **fixed} for s in seeds])


def evaluate_bio(adata, rep, name, label_key="cell_type", seed=SEED, resolutions=None):
    """Single-batch evaluation (PBMC3k): ARI, NMI, cell-type ASW, kNN macro-F1."""
    X, y = adata.obsm[rep], adata.obs[label_key].values
    return {"method": name, **cluster_scores(adata, name, label_key, seed, resolutions),
            "cell_type_ASW": (silhouette_score(X, y) + 1) / 2, "kNN_macroF1": knn_macro_f1(X, y, seed=seed)}


def summarize(per_seed):
    """Mean, sd and 'mean ± sd' tables over seeds, per method."""
    cols = [c for c in per_seed.columns if c not in ("method", "seed", "best_res")]
    g = per_seed.groupby("method")[cols]
    mean, sd = g.mean(), g.std(ddof=1)
    return mean.round(3), sd.round(3), mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)


def per_class_table(adata, reps, label_key="cell_type", seed=SEED):
    """Per-cell-type F1 (kNN and balanced LR, stratified CV) for each embedding; also returns the kNN predictions."""
    y = adata.obs[label_key].values
    labs = np.unique(y)
    cv = StratifiedKFold(n_splits=CFG["cv_folds"], shuffle=True, random_state=seed)
    rows, preds = {}, {}
    for m, rep in reps.items():
        X = adata.obsm[rep]
        preds[m] = cross_val_predict(KNeighborsClassifier(n_neighbors=CFG["knn_k"]), X, y, cv=cv)
        pl = cross_val_predict(_lr(), X, y, cv=cv, n_jobs=-1)
        rows[f"{m}_kNN"] = precision_recall_fscore_support(y, preds[m], labels=labs, zero_division=0)[2]
        rows[f"{m}_LR"] = precision_recall_fscore_support(y, pl, labels=labs, zero_division=0)[2]
    table = pd.DataFrame(rows, index=labs).round(2)
    table["n_cells"] = pd.Series(y).value_counts()
    return table, preds


def per_type_connectivity_mixing(adata, reps, graph_names, label_key="cell_type", batch_key="batch"):
    """Per cell type: graph connectivity and batch mixing for each embedding (what was preserved or lost)."""
    y, b = adata.obs[label_key].values, adata.obs[batch_key].values
    cts = sorted(set(y))
    conn, mix = {}, {}
    for m, rep in reps.items():
        A, X = adata.obsp[f"{graph_names[m]}_connectivities"], adata.obsm[rep]
        conn[m], mix[m] = {}, {}
        for ct in cts:
            idx = np.where(y == ct)[0]
            _, comp = connected_components(A[idx][:, idx], directed=False)
            conn[m][ct] = np.bincount(comp).max() / len(idx)
            nb = len(np.unique(b[idx]))
            mix[m][ct] = (1 - np.abs(silhouette_samples(X[idx], b[idx])).mean()) if nb >= 2 and len(idx) > nb else np.nan
    n_ct = pd.Series(y).value_counts()
    return (pd.DataFrame(conn).round(2).assign(n_cells=n_ct).sort_values("n_cells"),
            pd.DataFrame(mix).round(2).assign(n_cells=n_ct).sort_values("n_cells"))


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_umaps(adata, names, colors=("cell_type", "batch"), save=None):
    fig, axes = plt.subplots(len(colors), len(names), figsize=(4.5 * len(names), 4 * len(colors)), squeeze=False)
    for j, n in enumerate(names):
        for i, c in enumerate(colors):
            key = f"leiden_{n}" if c == "leiden" else c
            sc.pl.embedding(adata, basis=f"umap_{n}", color=key, ax=axes[i, j], show=False,
                            title=f"{n} | {c}", legend_loc="right margin" if j == len(names) - 1 else None)
    plt.tight_layout()
    if save:
        fig.savefig(f"{CFG['out_dir']}/{save}", dpi=150, bbox_inches="tight")
    plt.show()


def metric_bar(per_seed, metrics, order, save=None):
    g = per_seed.groupby("method")[metrics]
    mean, sd = g.mean(), g.std(ddof=1).fillna(0)
    x, w = np.arange(len(metrics)), 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 2, 4.5))
    for i, m in enumerate(order):
        ax.bar(x - 0.4 + w * (i + 0.5), mean.loc[m], w, yerr=sd.loc[m], capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (higher is better)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(f"{CFG['out_dir']}/{save}", dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------------------
# Geneformer zero-shot embeddings (own implementation of the rank-value encoding; the official
# package does not import under transformers 5.x)
# --------------------------------------------------------------------------------------
def build_gene_mapping(var_names, name2id, ens_map, tokdict, mediandict):
    """Gene symbol -> Ensembl ID (direct dictionary, then alias dictionary), restricted to the model
    vocabulary. Duplicate IDs are collapsed by summing counts (through the sparse matrix M: genes x vocab genes)."""
    genes = pd.Series(list(var_names), name="symbol")
    ens_direct = genes.map(name2id)
    ens = ens_direct.fillna(genes.str.upper().map(ens_map))
    in_vocab = ens.isin(set(tokdict))
    vocab_ens = sorted(set(ens[in_vocab]))
    index = {e: k for k, e in enumerate(vocab_ens)}
    rows = np.where(in_vocab.values)[0]
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, [index[e] for e in ens[in_vocab]])),
                      shape=(len(genes), len(vocab_ens)))
    tok_ids = np.array([tokdict[e] for e in vocab_ens])
    medians = np.array([mediandict.get(e, np.nan) for e in vocab_ens], dtype=np.float32)
    valid = np.isfinite(medians) & (medians > 0)
    return dict(genes=genes, ens=ens, ens_direct=ens_direct, in_vocab=in_vocab, M=M, index=index,
                tok_ids=tok_ids, medians=medians, valid=valid)


def rank_tokens(Cv, n_counts, tok_ids, medians, valid, cls_id, eos_id, max_len, target_sum=10_000):
    """Rank-value encoding: counts / n_counts * 1e4 / gene median, genes ranked in descending order,
    <cls> ... <eos>, truncated to max_len (cls and eos included)."""
    Cv, out = Cv.tocsr(), []
    for i in range(Cv.shape[0]):
        s, e = Cv.indptr[i], Cv.indptr[i + 1]
        idx, val = Cv.indices[s:e], Cv.data[s:e]
        keep = (val > 0) & valid[idx]
        idx, val = idx[keep], val[keep]
        order = np.argsort(-(val / n_counts[i] * target_sum / medians[idx]))
        out.append(np.concatenate([[cls_id], tok_ids[idx[order]][: max_len - 2], [eos_id]]).astype(np.int64))
    return out


def embed_tokens(model, tokens, pad_id, eos_id, bs=8, layer=-2, device="cuda"):
    """Cell embedding = mean over gene tokens (no pad/cls/eos) of the second-to-last layer
    (hidden_states[-1] is the last layer). Cells are processed in length order to limit padding."""
    import torch
    n, lens_ = len(tokens), np.array([len(t) for t in tokens])
    order = np.argsort(lens_)
    out = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, n, bs):
            idx = order[k:k + bs]
            L = lens_[idx].max()
            ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :lens_[i]] = torch.from_numpy(tokens[i])
            ids = ids.to(device)
            att = (ids != pad_id).long()
            with torch.autocast("cuda", dtype=torch.float16):
                hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer]
            gm = ((ids > eos_id) & (att == 1)).unsqueeze(-1)       # gene tokens only (special ids are 0..3)
            out[idx] = ((hs.float() * gm).sum(1) / gm.sum(1).clamp(min=1)).cpu().numpy()
    return out
