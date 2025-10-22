# Ben Kabongo
# October 2025


import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional

# pip install vector-quantize-pytorch
from vector_quantize_pytorch import ResidualVQ


# ---------------------- Matplotlib style ----------------------

def set_mpl_style():
    mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        "#8da0cb", "#fc8d62", "#66c2a5", "#e78ac3", "#a6d854"
    ])
    mpl.rcParams['grid.linestyle'] = ":"
    mpl.rcParams['grid.linewidth'] = 0.6
    mpl.rcParams['grid.alpha'] = 0.7
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

def stage_colors():
    return mpl.rcParams['axes.prop_cycle'].by_key()['color'][:3]


# ---------------------- NumPy utilities ----------------------

def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def cosine_mat_np(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return X @ C.T


# ---------------------- IO helpers ----------------------

def load_embeddings(path: str) -> np.ndarray:
    x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise ValueError("embedding_path must be a torch.Tensor.")
    x = x.detach().cpu().float()
    x = F.normalize(x, p=2, dim=1)
    return x.numpy()

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)


# ---------------------- Codebooks helpers ----------------------

def get_stage_codebooks(quantizer) -> List[np.ndarray]:
    """
    Returns [C0, C1, C2, ...] with each Ck shape (K_k, D) as float32 (L2-normalized).
    """
    Cbooks = quantizer.codebooks
    centers = []
    if isinstance(Cbooks, torch.Tensor):  # uniform K across layers -> [Q, K, D]
        for q in range(Cbooks.shape[0]):
            Cq = Cbooks[q].detach().cpu().float().numpy()
            Cq = Cq / (np.linalg.norm(Cq, axis=1, keepdims=True) + 1e-12)
            centers.append(Cq)
    else:  # tuple of [K_q, D]
        for Cq in Cbooks:
            Cq = Cq.detach().cpu().float().numpy()
            Cq = Cq / (np.linalg.norm(Cq, axis=1, keepdims=True) + 1e-12)
            centers.append(Cq)
    return centers

def save_codebooks_pt(centers_all: List[np.ndarray], path: str):
    """Save stage codebooks to a single .pt (Torch) file."""
    payload = {
        "stages": len(centers_all),
        "centers": [torch.from_numpy(C.astype(np.float32)) for C in centers_all]
    }
    torch.save(payload, path)

def load_codebooks_pt(path: str) -> List[np.ndarray]:
    """Load stage codebooks saved by save_codebooks_pt()."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "centers" in obj:
        Cs = [c.detach().cpu().float().numpy() for c in obj["centers"]]
    else:
        Cs = [c.detach().cpu().float().numpy() if isinstance(c, torch.Tensor) else np.asarray(c, dtype=np.float32)
              for c in obj]
    # L2-normalize for safety
    Cs = [c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12) for c in Cs]
    return Cs

def assign_codes_to_embedding(x: np.ndarray, centers_all: List[np.ndarray]) -> Dict[str, object]:
    """
    Given a single embedding x (shape (d,) or (1,d)), greedily assign codes per stage
    using residual selection; report per-stage cumulative cosines to x.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    x = l2_normalize_np(x)

    codes, cum_cos = [], []
    cum = np.zeros_like(x, dtype=np.float32)
    for s, Cs in enumerate(centers_all):
        resid = x - cum
        resid = l2_normalize_np(resid)
        sims = resid @ Cs.T
        idx = int(np.argmax(sims[0]))
        codes.append(idx)

        cum = cum + Cs[idx][None, :]
        cum_n = l2_normalize_np(cum)
        cos_val = float((x * cum_n).sum(axis=1)[0])
        cum_cos.append(cos_val)

    return {"codes": codes, "cum_cos": cum_cos}


# ---------- Cumulative reconstruction + cosine at each stage ----------

def cumulative_recon_and_cosines(
    X: np.ndarray,
    ids: np.ndarray,
    centers_all: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each stage s (1..S), build the cumulative reconstruction
        recon_s[i] = sum_{t=1..s} e_t[ code_t(i) ]
    and compute cumulative cosine:
        cos_s[i] = cosine( x_i , recon_s[i] )
    Returns (recons_by_stage, cosines_by_stage).
    """
    Xn = l2_normalize_np(X)
    S = min(ids.shape[1], len(centers_all))
    recons, cosines = [], []

    cum = np.zeros_like(X, dtype=np.float32)
    for s in range(S):
        Ct = centers_all[s]                                # (K_s, d)
        ct_indices = np.clip(ids[:, s].astype(int), 0, Ct.shape[0] - 1)
        cum = cum + Ct[ct_indices]                        # cumulative sum
        cum_n = l2_normalize_np(cum)
        cos_s = (Xn * cum_n).sum(axis=1)
        recons.append(cum.copy())
        cosines.append(cos_s)
    return recons, cosines


# ---------------------- Representative selection (path-consistent) -----------------------

def select_reps_stage1(
    X: np.ndarray, 
    ids: np.ndarray, 
    C1: np.ndarray,
    df_group: pd.DataFrame, 
    strategy: str = "nearest"
) -> pd.DataFrame:
    """
    Stage-1 selection (not used directly in the final path-consistent pipeline, kept for reference):
      - 'freq': highest frequency inside each code-1
      - 'nearest': nearest by residual-aware cosine: cosine(x, e1[c1])  (selection only)
    """
    Xn = l2_normalize_np(X)
    sims = cosine_mat_np(Xn, l2_normalize_np(C1))        # (n, K1)

    rows = []
    labels = ids[:, 0].astype(int)
    for c1 in np.unique(labels):
        idx = np.where(labels == c1)[0]
        if len(idx) == 0:
            continue

        if strategy == "freq":
            j = idx[np.argmax(df_group.iloc[idx]["frequency"].values)]
        else:
            j = idx[np.argmax(sims[idx, c1])]

        rows.append({
            "stage": 1,
            "code1": int(c1),
            "local_index": int(j),
            "global_index": int(df_group.index.values[j]),
            "statement": str(df_group.iloc[j]["statement"]),
            "frequency": float(df_group.iloc[j]["frequency"])
        })
    return pd.DataFrame(rows)


def select_reps_stage2(
    X: np.ndarray, 
    ids: np.ndarray, 
    C1: np.ndarray, 
    C2: np.ndarray,
    df_group: pd.DataFrame, 
    strategy: str = "nearest"
) -> pd.DataFrame:
    """
    Stage-2 selection (reference):
      - 'freq': highest frequency inside each (code1, code2)
      - 'nearest': residual-aware cosine to e2[c2] (selection only)
    """
    l1, l2 = ids[:, 0].astype(int), ids[:, 1].astype(int)
    e1 = C1[l1]
    r1 = l2_normalize_np(X - e1)
    sims2 = cosine_mat_np(r1, l2_normalize_np(C2))

    rows = []
    for c1 in np.unique(l1):
        idx1 = np.where(l1 == c1)[0]
        if len(idx1) == 0:
            continue
        sub_l2 = l2[idx1]
        for c2 in np.unique(sub_l2):
            idx12 = idx1[sub_l2 == c2]
            if len(idx12) == 0:
                continue
            if strategy == "freq":
                pick = idx12[np.argmax(df_group.iloc[idx12]["frequency"].values)]
            else:
                pick = idx12[np.argmax(sims2[idx12, c2])]
            rows.append({
                "stage": 2,
                "code1": int(c1),
                "code2": int(c2),
                "local_index": int(pick),
                "global_index": int(df_group.index.values[pick]),
                "statement": str(df_group.iloc[pick]["statement"]),
                "frequency": float(df_group.iloc[pick]["frequency"])
            })
    return pd.DataFrame(rows)


def select_reps_stage3(
    X: np.ndarray, 
    ids: np.ndarray, 
    C1: np.ndarray, 
    C2: np.ndarray, 
    C3: np.ndarray,
    df_group: pd.DataFrame, 
    strategy: str = "nearest"
) -> pd.DataFrame:
    """
    Stage-3 selection (reference):
      - 'freq' or 'nearest' (residual-aware to e3[c3]) – selection only
    """
    l1, l2, l3 = ids[:, 0].astype(int), ids[:, 1].astype(int), ids[:, 2].astype(int)
    e1, e2 = C1[l1], C2[l2]
    r2 = l2_normalize_np(X - e1 - e2)
    sims3 = cosine_mat_np(r2, l2_normalize_np(C3))

    rows = []
    parents = np.unique(np.stack([l1, l2], axis=1), axis=0)
    for c1, c2 in parents:
        idx12 = np.where((l1 == c1) & (l2 == c2))[0]
        if len(idx12) == 0:
            continue
        sub_l3 = l3[idx12]
        for c3 in np.unique(sub_l3):
            idx123 = idx12[sub_l3 == c3]
            if len(idx123) == 0:
                continue
            if strategy == "freq":
                pick = idx123[np.argmax(df_group.iloc[idx123]["frequency"].values)]
            else:
                pick = idx123[np.argmax(sims3[idx123, c3])]
            rows.append({
                "stage": 3,
                "code1": int(c1),
                "code2": int(c2),
                "code3": int(c3),
                "local_index": int(pick),
                "global_index": int(df_group.index.values[pick]),
                "statement": str(df_group.iloc[pick]["statement"]),
                "frequency": float(df_group.iloc[pick]["frequency"])
            })
    return pd.DataFrame(rows)


# ---------------------- Base table + path-consistent selection ----------------------

def make_base_table(
    X: np.ndarray,
    ids: np.ndarray,
    centers_all: List[np.ndarray],
    df_group: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a row-per-example table with codes, cumulative cosines per stage,
    and the fields we need to rank + plot.
    """
    _, cos_by_stage = cumulative_recon_and_cosines(X, ids, centers_all)

    base = pd.DataFrame({
        "local_index": np.arange(len(X)),
        "global_index": df_group.index.values,
        "statement": df_group["statement"].astype(str).values,
        "frequency": df_group["frequency"].astype(float).values,
        "c1": ids[:, 0].astype(int),
        "cos1": cos_by_stage[0],
    })

    if len(centers_all) >= 2:
        base["c2"] = ids[:, 1].astype(int)
        base["cos2"] = cos_by_stage[1]
    else:
        base["c2"] = -1
        base["cos2"] = np.nan

    if len(centers_all) >= 3:
        base["c3"] = ids[:, 2].astype(int)
        base["cos3"] = cos_by_stage[2]
    else:
        base["c3"] = -1
        base["cos3"] = np.nan

    return base


def select_path_consistent_reps(
    base: pd.DataFrame,
    *,
    strategy: str = "nearest",          # 'nearest' (use cos) or 'freq'
    min_count: int = 10,                # only keep paths with ≥ min_count members
    leaf_cap: int = 3,                  # how many statements per leaf (c1,c2,c3)
    s2_cap_per_parent: int = 20,        # cap per (c1) panel when aggregating s2 reps
    s1_cap_per_parent: int = 20         # cap per c1 panel when aggregating s1 reps
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1) Select at the leaf (deepest) level.
    2) Propagate upward so selections overlap across stages.

    Returns (reps_stage1, reps_stage2, reps_stage3).
    """
    have_s2 = "c2" in base.columns and base["c2"].min() >= 0
    have_s3 = "c3" in base.columns and base["c3"].min() >= 0

    # -------- Stage 3: select within (c1,c2,c3)
    reps3_rows = []
    if have_s3:
        for (c1, c2, c3), sub in base.groupby(["c1", "c2", "c3"], sort=True):
            if len(sub) < min_count:
                continue
            if strategy == "freq":
                pick = sub.sort_values(["frequency", "cos3"], ascending=False).head(leaf_cap)
            else:
                pick = sub.sort_values(["cos3", "frequency"], ascending=False).head(leaf_cap)
            for _, r in pick.iterrows():
                reps3_rows.append({
                    "stage": 3,
                    "code1": int(c1),
                    "code2": int(c2),
                    "code3": int(c3),
                    "local_index": int(r.local_index),
                    "global_index": int(r.global_index),
                    "statement": r.statement,
                    "frequency": float(r.frequency),
                    "cos1": float(r.cos1),
                    "cos2": float(r.cos2),
                    "cos3": float(r.cos3)
                })
    reps3 = pd.DataFrame(reps3_rows)

    # -------- Stage 2: union of selected Stage-3 rows within (c1,c2)
    reps2 = pd.DataFrame(columns=[
        "stage","code1","code2","code3","local_index","global_index","statement","frequency","cos1","cos2"
    ])
    if have_s2 and not reps3.empty:
        reps2 = (
            reps3
            .sort_values(["cos2","frequency"], ascending=False)
            .groupby(["code1","code2"], as_index=False)
            .head(s2_cap_per_parent)
            .copy()
        )
        reps2["stage"] = 2
        # keep code3 so we can order by common codes
        reps2 = reps2[["stage","code1","code2","code3","local_index","global_index",
                       "statement","frequency","cos1","cos2"]]

    # -------- Stage 1: union of selected Stage-2 rows within c1
    reps1 = pd.DataFrame(columns=[
        "stage","code1","code2","code3","local_index","global_index","statement","frequency","cos1"
    ])
    if not reps2.empty:
        reps1 = (
            reps2
            .sort_values(["cos1","frequency"], ascending=False)
            .groupby(["code1"], as_index=False)
            .head(s1_cap_per_parent)
            .copy()
        )
        reps1["stage"] = 1
        reps1 = reps1[["stage","code1","code2","code3","local_index","global_index",
                       "statement","frequency","cos1"]]

    return reps1, reps2, reps3


# ---------------------- Bar plots for representatives (cumulative cosine) ----------------------

def _truncate(s: str, n: int = 70) -> str:
    return s if len(s) <= n else (s[: n - 1] + "…")

def _code_tag(row: pd.Series) -> str:
    # (c1, *, *), (c1, c2, *), (c1, c2, c3)
    if row["stage"] == 1:
        return f"({row['code1']}, *, *)"
    if row["stage"] == 2:
        return f"({row['code1']}, {row['code2']}, *)"
    return f"({row['code1']}, {row['code2']}, {row['code3']})"

def plot_reps_bars_cumulative(
    reps_df: pd.DataFrame,
    cum_cos: np.ndarray,
    ids: np.ndarray,
    out_path: str,
    title: str,
    stage_index: int,
    global_to_local: Optional[Dict[int, int]] = None,
    max_bars: int = 20
):
    """
    Horizontal bars:
      x = cumulative cosine at current stage for the picked statement,
      y = statement text (truncated), with a right-aligned code tag.

    Ordering: by common codes first (code1,code2,code3), then by cosine desc.
    """
    if reps_df.empty:
        return

    reps = reps_df.copy()

    if "local_index" in reps.columns:
        loc_idx = reps["local_index"].to_numpy()
    else:
        assert global_to_local is not None, "global_to_local mapping required when local_index is absent"
        loc_idx = np.array([global_to_local[g] for g in reps["global_index"].to_numpy()])

    reps["cosine_to_code"] = cum_cos[loc_idx]
    reps["tag"] = reps.apply(_code_tag, axis=1)

    # Ensure code2/code3 exist (fill with -1 if missing)
    for col in ("code2", "code3"):
        if col not in reps.columns:
            reps[col] = -1

    # Sort by codes, then cosine desc
    reps = reps.sort_values(by=["code1", "code2", "code3", "cosine_to_code"],
                            ascending=[True, True, True, False])

    # Cap number of bars
    reps = reps.head(max_bars)

    y_pos = np.arange(len(reps))
    labels = reps["statement"].apply(lambda s: _truncate(str(s))).values
    vals = reps["cosine_to_code"].values
    tags = reps["tag"].values

    plt.figure(figsize=(12, max(4, 0.45 * len(vals))))
    plt.barh(y_pos, vals, color=stage_colors()[stage_index - 1])
    plt.yticks(y_pos, labels)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Statements")
    plt.title(title)
    # right-aligned code tags
    for y, x, tag in zip(y_pos, vals, tags):
        plt.text(min(x + 0.01, 0.99), y, tag, va="center", ha="left", fontsize=10)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------- Intra cumulative cosine “box” plots ----------------------

def plot_intra_cumulative_cosine_box(
    cos_vals: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
    ylabel: str
):
    """
    Boxplot of cumulative cosine grouped by the code(s) at the current stage.
    """
    if cos_vals.size == 0 or labels.size == 0:
        return
    uniq = sorted(np.unique(labels))
    parts = [cos_vals[labels == c] for c in uniq if np.any(labels == c)]
    if not parts:
        return
    plt.figure(figsize=(10, 5))
    plt.boxplot(parts, labels=[f"c{c}" for c in uniq], showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------- Training ----------------------

@dataclass
class RVQConfig:
    codebook_size: int = 32
    num_quantizers: int = 3
    use_cosine_sim: bool = True
    shared_codebook: bool = True
    stochastic_sample_codes: bool = True
    kmeans_init: bool = True
    kmeans_iters: int = 10
    threshold_ema_dead_code: int = 2
    rotation_trick: bool = True

@dataclass
class LoopConfig:
    epochs: int = 5
    batch_size: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def train_vq_group(X: np.ndarray, vq_cfg: RVQConfig, loop: LoopConfig) -> Dict:
    torch.manual_seed(loop.seed)
    np.random.seed(loop.seed)

    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=loop.device)

    quantizer = ResidualVQ(
        dim=d,
        codebook_size=vq_cfg.codebook_size,
        num_quantizers=vq_cfg.num_quantizers,
        use_cosine_sim=vq_cfg.use_cosine_sim,
        shared_codebook=vq_cfg.shared_codebook,
        stochastic_sample_codes=vq_cfg.stochastic_sample_codes,
        kmeans_init=vq_cfg.kmeans_init,
        kmeans_iters=vq_cfg.kmeans_iters,
        threshold_ema_dead_code=vq_cfg.threshold_ema_dead_code,
        rotation_trick=vq_cfg.rotation_trick,
    ).to(loop.device)
    quantizer.train()

    # EMA updates happen inside forward; no grads needed
    for ep in range(loop.epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, loop.batch_size):
            idx = perm[i: i + loop.batch_size]
            xb = X_t[idx]
            with torch.no_grad():
                z_q, _, _ = quantizer(xb)
            if (i // loop.batch_size) % 50 == 0:
                cos = F.cosine_similarity(xb, z_q, dim=1).mean().item()
                print(f"    epoch {ep+1} | batch {(i//loop.batch_size)+1} | mean cos≈{cos:.4f}")

    # Collect assignments and z_q
    quantizer.eval()
    with torch.no_grad():
        zq_full, ids_full, _ = quantizer(X_t)      # (N, d), (N, L)
        zq_full = F.normalize(zq_full, p=2, dim=1)
        cos_full = F.cosine_similarity(X_t, zq_full, dim=1).cpu().numpy()

    ids_full = ids_full.long().cpu().numpy()
    zq_full = zq_full.cpu().numpy()

    # Stage-1 usage
    stage1 = ids_full[:, 0].astype(np.int64)
    K = vq_cfg.codebook_size
    stage1 = np.clip(stage1, 0, K - 1)
    usage = np.bincount(stage1, minlength=K)
    dead = int((usage == 0).sum())

    centers_all = get_stage_codebooks(quantizer)  # List[np.ndarray]

    return dict(
        quantizer=quantizer,
        recon=zq_full,
        cos=cos_full,
        ids=ids_full,
        stage1=stage1,
        usage=usage,
        dead_codes=dead,
        centers_stage_all=centers_all
    )


# ---------------------- Simple plots ----------------------

def plot_code_usage(usage: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(usage)), usage, width=1.0)
    plt.xlabel("Code id"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_distortion_hist(cos: np.ndarray, out_path: str, title: str):
    dist = 1.0 - cos
    plt.figure(figsize=(6, 4))
    plt.hist(dist, bins=50)
    plt.xlabel("Distortion = 1 - cos"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ---------------------- Main ----------------------

def main(args):
    set_mpl_style()
    out_root = args.output_dir
    ensure_dirs(out_root)
    groups_root = os.path.join(out_root, "groups"); os.makedirs(groups_root, exist_ok=True)

    # Load data
    print("[1/5] Loading data ...")
    X_all = load_embeddings(args.embedding_path)   # (N, d)
    N, d = X_all.shape
    df = pd.read_csv(args.sts_csv)
    if len(df) != N:
        raise ValueError(f"Row mismatch: embeddings N={N}, CSV rows={len(df)}.")
    for col in ["statement", "topic", "sentiment", "frequency"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in sts_csv.")
    df["frequency"] = df["frequency"].astype(float).fillna(1.0)

    vq_cfg = RVQConfig(codebook_size=args.codebook_size, num_quantizers=args.num_quantizers)
    loop = LoopConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    global_rows, summaries = [], []
    common_prompt_entries: List[Dict[str, Optional[str]]] = []

    print("[2/5] Training VQ per group ...")
    groups = df.groupby(["topic", "sentiment"], dropna=False, sort=True)
    for (topic, sentiment), g in groups:
        idx = g.index.values
        global_to_local = {gi: i for i, gi in enumerate(idx)}
        Xg = X_all[idx]
        n = len(idx)
        tag = f"{topic}__{sentiment}"
        out_dir = os.path.join(groups_root, tag)
        ensure_dirs(out_dir)

        print(f"  • Group (topic='{topic}', sentiment='{sentiment}') | n={n}")

        # Safety: need enough samples to populate codebook
        if n < max(200, args.codebook_size):
            print("    [warn] group too small for reliable codebook, skipping.")
            pd.DataFrame({"global_index": idx, "stage1": -1, "cosine_vq": np.nan}).to_csv(
                os.path.join(out_dir, "vq_per_point.csv"), index=False
            )
            with open(os.path.join(out_dir, "vq_meta.json"), "w") as f:
                json.dump({"topic": str(topic), "sentiment": str(sentiment),
                           "n_points": int(n), "skipped": True}, f, indent=2)
            continue

        # Train VQ (EMA)
        out = train_vq_group(Xg, vq_cfg, loop)

        stage1 = out["stage1"]
        cos = out["cos"]
        ids_all = out["ids"]
        centers_all = out["centers_stage_all"]      # list of (K, D)
        num_stages = len(centers_all)

        # Save codebooks (.pt) for this group
        save_codebooks_pt(centers_all, os.path.join(out_dir, "codebooks_stage_all.pt"))

        # Per-point CSV
        per_point = pd.DataFrame({"global_index": idx, "stage1": ids_all[:, 0].astype(int)})
        for l in range(ids_all.shape[1]):
            per_point[f"stage_{l}"] = ids_all[:, l].astype(int)
        per_point["cosine_vq"] = cos.astype(float)
        per_point.to_csv(os.path.join(out_dir, "vq_per_point.csv"), index=False)

        # Cumulative recon + cosines at stage 1..S
        _, cos_by_stage = cumulative_recon_and_cosines(Xg, ids_all, centers_all)

        # Build base with cumulative cosines (cos1, cos2, cos3)
        base = make_base_table(Xg, ids_all, centers_all, g)

        # Path-consistent reps (leaf -> parents)
        reps1, reps2, reps3 = select_path_consistent_reps(
            base,
            strategy=args.rep_strategy,
            min_count=10,            # keep only sufficiently populated codes/paths
            leaf_cap=3,              # per (c1,c2,c3)
            s2_cap_per_parent=20,
            s1_cap_per_parent=20
        )

        if not reps1.empty: reps1.to_csv(os.path.join(out_dir, "reps_stage1.csv"), index=False)
        if not reps2.empty: reps2.to_csv(os.path.join(out_dir, "reps_stage2.csv"), index=False)
        if not reps3.empty: reps3.to_csv(os.path.join(out_dir, "reps_stage3.csv"), index=False)

        # BAR PLOTS (cumulative cosine at the corresponding stage)
        # Stage 1
        if not reps1.empty:
            plot_reps_bars_cumulative(
                reps1,
                cum_cos=cos_by_stage[0],
                ids=ids_all,
                out_path=os.path.join(out_dir, "figures", "reps_stage1_bars.pdf"),
                title=f"{topic}, {sentiment}: stage 1",
                stage_index=1,
                global_to_local=global_to_local,
                max_bars=args.max_reps_per_plot
            )

        # Stage 2 — one figure per parent c1
        if num_stages >= 2 and not reps2.empty:
            for c1 in sorted(reps2["code1"].unique()):
                sub = reps2[reps2["code1"] == c1].copy()
                if sub.empty:
                    continue
                plot_reps_bars_cumulative(
                    sub,
                    cum_cos=cos_by_stage[1],
                    ids=ids_all,
                    out_path=os.path.join(out_dir, "figures", f"reps_stage2_parent_{c1}.pdf"),
                    title=f"{topic}, {sentiment}: stage 2 (c1={c1})",
                    stage_index=2,
                    global_to_local=global_to_local,
                    max_bars=args.max_reps_per_plot
                )

        # Stage 3 — one figure per (c1,c2) parent
        if num_stages >= 3 and not reps3.empty:
            for (c1, c2), sub in reps3.groupby(["code1", "code2"]):
                plot_reps_bars_cumulative(
                    sub,
                    cum_cos=cos_by_stage[2],
                    ids=ids_all,
                    out_path=os.path.join(out_dir, "figures", f"reps_stage3_parent_{c1}_{c2}.pdf"),
                    title=f"{topic}, {sentiment}: stage 3 ({c1},{c2})",
                    stage_index=3,
                    global_to_local=global_to_local,
                    max_bars=args.max_reps_per_plot
                )

        # Usage + distortion plots (global for the group)
        usage = out["usage"]; dead = out["dead_codes"]
        plot_code_usage(usage, os.path.join(out_dir, "figures", "stage1_usage_hist.pdf"),
                        title=f"{topic}, {sentiment}: Stage 1 usage")
        plot_distortion_hist(cos, os.path.join(out_dir, "figures", "distortion_hist.pdf"),
                             title=f"{topic}, {sentiment}: Distortion")

        # Intra cumulative-cosine boxes at s=1..S (ALWAYS generate, independent of reps)
        plot_intra_cumulative_cosine_box(
            cos_vals=cos_by_stage[0],
            labels=ids_all[:, 0].astype(int),
            out_path=os.path.join(out_dir, "figures", "intra_cumcos_stage1_box.pdf"),
            title=f"{topic}, {sentiment}: stage 1",
            ylabel="Cos"
        )
        if num_stages >= 2:
            plot_intra_cumulative_cosine_box(
                cos_vals=cos_by_stage[1],
                labels=ids_all[:, 1].astype(int),
                out_path=os.path.join(out_dir, "figures", "intra_cumcos_stage2_box.pdf"),
                title=f"{topic}, {sentiment}: stage 2",
                ylabel="Cos"
            )
        if num_stages >= 3:
            plot_intra_cumulative_cosine_box(
                cos_vals=cos_by_stage[2],
                labels=ids_all[:, 2].astype(int),
                out_path=os.path.join(out_dir, "figures", "intra_cumcos_stage3_box.pdf"),
                title=f"{topic}, {sentiment}: stage 3",
                ylabel="Cos"
            )

        # Add to common (non-rep) plots prompt
        common_prompt_entries.append({
            "topic": str(topic), "sentiment": str(sentiment),
            "usage": os.path.join(out_dir, "figures", "stage1_usage_hist.pdf"),
            "distortion": os.path.join(out_dir, "figures", "distortion_hist.pdf"),
            "box_s1": os.path.join(out_dir, "figures", "intra_cumcos_stage1_box.pdf"),
            "box_s2": (os.path.join(out_dir, "figures", "intra_cumcos_stage2_box.pdf") if num_stages >= 2 else None),
            "box_s3": (os.path.join(out_dir, "figures", "intra_cumcos_stage3_box.pdf") if num_stages >= 3 else None),
        })

        # Group summary
        dist = 1.0 - cos
        valid = np.isfinite(dist)
        dist_stats = {
            "mean": float(np.mean(dist[valid])) if valid.any() else None,
            "p50": float(np.median(dist[valid])) if valid.any() else None,
            "p90": float(np.percentile(dist[valid], 90)) if valid.any() else None,
        }
        sil = None
        uniq = np.unique(stage1)
        if uniq.size >= 2 and n > 1000:
            try:
                sil = float(silhouette_score(l2_normalize_np(Xg), stage1, metric="cosine"))
            except Exception:
                sil = None

        # quick per-stage dead/usage summary
        stage_summ = {}
        for s in range(min(3, num_stages)):
            labels_s = ids_all[:, s].astype(int)
            K_s = centers_all[s].shape[0]
            usage_s = np.bincount(np.clip(labels_s, 0, K_s-1), minlength=K_s)
            stage_summ[f"stage{s+1}_usage_nonzero"] = int((usage_s > 0).sum())
            stage_summ[f"stage{s+1}_dead_codes"]    = int((usage_s == 0).sum())
            stage_summ[f"stage{s+1}_dead_ratio"]    = float((usage_s == 0).mean())

        meta = {
            "topic": str(topic), "sentiment": str(sentiment),
            "n_points": int(n),
            "codebook_size": int(args.codebook_size),
            "num_quantizers": int(args.num_quantizers),
            "dead_codes_stage1": int(dead),
            "dead_codes_ratio_stage1": float(dead / args.codebook_size),
            "usage_nonzero_stage1": int((usage > 0).sum()),
            "distortion_mean": dist_stats["mean"],
            "distortion_p50": dist_stats["p50"],
            "distortion_p90": dist_stats["p90"],
            "silhouette_cosine_stage1": sil,
            **stage_summ
        }
        with open(os.path.join(out_dir, "vq_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Accumulate globals
        tmp = per_point[["global_index", "stage1", "cosine_vq"]].copy()
        tmp["topic"] = str(topic); tmp["sentiment"] = str(sentiment)
        global_rows.append(tmp)
        summaries.append({**meta, "mean_cosine_vq": float(np.mean(cos)) if np.isfinite(cos).any() else None})

        print(f"    saved reps & cumulative-cosine figures for stages 1..{min(3, num_stages)}.")

    # Global artifacts
    print("[3/5] Writing global artifacts ...")
    if global_rows:
        gpp = pd.concat(global_rows, ignore_index=True)
        gpp = gpp[["global_index", "topic", "sentiment", "stage1", "cosine_vq"]]
        gpp.to_csv(os.path.join(out_root, "global_vq_per_point.csv"), index=False)
    gsum = pd.DataFrame(summaries)
    gsum.to_csv(os.path.join(out_root, "global_vq_group_summary.csv"), index=False)

    # Global aggregates: mean and std over numeric columns
    num_cols = gsum.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0 and len(gsum) > 0:
        agg_mean = gsum[num_cols].mean(numeric_only=True)
        agg_std  = gsum[num_cols].std(ddof=1, numeric_only=True)

        agg_df = pd.DataFrame({
            "metric": num_cols,
            "mean": [agg_mean[c] for c in num_cols],
            "std":  [agg_std[c] for c in num_cols]
        })
        agg_df.to_csv(os.path.join(out_root, "global_stats_mean_std.csv"), index=False)

        with open(os.path.join(out_root, "global_stats_mean_std.json"), "w") as f:
            json.dump({m: {"mean": float(agg_mean[m]), "std": float(agg_std[m])} for m in num_cols},
                      f, indent=2)
        print(" - global_stats_mean_std.csv")

    # Global plots
    print("[4/5] Global plots ...")
    try:
        plt.figure(figsize=(7, 4))
        plt.hist(gsum["dead_codes_ratio_stage1"].fillna(0.0), bins=30)
        plt.xlabel("Dead code ratio (stage-1)"); plt.ylabel("Count of groups")
        plt.title("Dead code ratio across groups"); plt.tight_layout()
        plt.savefig(os.path.join(out_root, "global_dead_code_ratio_hist.pdf")); plt.close()

        val = gsum["mean_cosine_vq"].dropna()
        if len(val) > 0:
            plt.figure(figsize=(7, 4))
            plt.hist(val, bins=30)
            plt.xlabel("Mean cos per group"); plt.ylabel("Count of groups")
            plt.title("Mean reconstruction cosine across groups"); plt.tight_layout()
            plt.savefig(os.path.join(out_root, "global_mean_cosine_hist.pdf")); plt.close()
    except Exception as e:
        print(f"[WARN] Global plots failed: {e}")

    # Common prompt file for NON-rep plots across all groups
    common_path = os.path.join(out_root, "figures", "_common_plots_prompt.txt")
    os.makedirs(os.path.dirname(common_path), exist_ok=True)
    with open(common_path, "w", encoding="utf-8") as f:
        f.write("Assemble per-group panels (topic × sentiment) with the following figures.\n")
        f.write("For each group, include: Stage-1 Usage, Distortion, and Intra Cumulative Cosine for stages 1..S.\n\n")
        for item in common_prompt_entries:
            f.write(f"- Group: {item['topic']} — {item['sentiment']}\n")
            f.write(f"  • Usage: {item['usage']}\n")
            f.write(f"  • Distortion: {item['distortion']}\n")
            if item.get("box_s1"): f.write(f"  • Intra cum-cos (S1): {item['box_s1']}\n")
            if item.get("box_s2"): f.write(f"  • Intra cum-cos (S2): {item['box_s2']}\n")
            if item.get("box_s3"): f.write(f"  • Intra cum-cos (S3): {item['box_s3']}\n")
            f.write("\n")
    print(f"[info] Wrote common plots prompt: {common_path}")

    print("\nDone.")
    print(f"Artifacts root: {out_root}")
    print("Global files:")
    print(" - global_vq_per_point.csv")
    print(" - global_vq_group_summary.csv")
    print(" - global_stats_mean_std.csv")
    print(" - global_dead_code_ratio_hist.pdf")
    print(" - global_mean_cosine_hist.pdf")
    print(" - figures/_common_plots_prompt.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual VQ per (topic × sentiment), cumulative-cosine plots")
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--sts_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # VQ capacity
    parser.add_argument("--codebook_size", type=int, default=32)
    parser.add_argument("--num_quantizers", type=int, default=3)

    # Loop
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    # Rep selection and plotting
    parser.add_argument("--rep_strategy", type=str, default="nearest", choices=["nearest", "freq"],
        help="Leaf selection per (c1,c2,c3): nearest (by residual-aware cosine) or freq (highest frequency)")
    parser.add_argument("--max_reps_per_plot", type=int, default=20,
        help="Max number of statements per representatives plot")

    args = parser.parse_args()
    main(args)