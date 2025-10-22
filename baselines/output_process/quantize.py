# Ben Kabongo
# October 2025


import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def l2_normalize_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(dim=dim, keepdim=True), min=eps)

def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def read_sts_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    for col in ("statement", "topic", "sentiment"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in STS dataframe.")
    return df

def load_embeddings_pt(path: str, device: torch.device) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Embeddings file must contain a torch.Tensor. Got: {type(obj)}")
    X = obj.to(device=device, dtype=torch.float32).contiguous()
    if X.dim() != 2:
        raise ValueError(f"Expected 2D embeddings tensor (N, d); got shape {tuple(X.shape)}")
    return X

def load_codebooks_pt(path: str) -> List[np.ndarray]:
    """
    Load stage codebooks saved as either:
      - dict with key "centers": list of tensors/arrays
      - list/tuple of tensors/arrays
    Returns a list of L2-normalized numpy arrays [(K_s, d), ...].
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "centers" in obj:
        Cs_list = [c.detach().cpu().float().numpy() for c in obj["centers"]]
    else:
        Cs_list = [
            c.detach().cpu().float().numpy() if isinstance(c, torch.Tensor)
            else np.asarray(c, dtype=np.float32)
            for c in obj
        ]
    Cs_list = [l2_normalize_np(C, axis=1) for C in Cs_list]
    return Cs_list

def to_torch_codebooks(
    centers_all_np: List[np.ndarray],
    device: torch.device,
) -> List[torch.Tensor]:
    """Convert normalized numpy codebooks to torch tensors"""
    out = []
    for C in centers_all_np:
        t = torch.from_numpy(C).to(device=device, dtype=torch.float32)
        # already L2-normalized row-wise
        out.append(t)
    return out

def scan_codebooks_root(root: str) -> Dict[Tuple[str, str], Path]:
    """
    Return mapping {(topic, sentiment) -> path_to_folder}.
    Expects folders named 'topic__sentiment' directly under root.
    """
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"Codebooks root not found: {root}")
    mapping: Dict[Tuple[str, str], Path] = {}
    for child in root_p.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("__")
        if len(parts) != 2:
            continue
        t, s = parts
        mapping[(t, s)] = child
    return mapping


@torch.no_grad()
def assign_codes_batchwise(
    X: torch.Tensor,                      # (B, d)
    centers_all: List[torch.Tensor],      # list of (K_s, d)
    normalize_inputs: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    For each vector x_b, at each stage s:
      - compute resid = normalize(x - cum)
      - pick center with max cosine(resid, center)
      - update cum += chosen_center
      - record cumulative cosine cos(x, normalize(cum))
    Returns:
      - codes: LongTensor (B, S)
      - cum_cos: FloatTensor (B, S)
    """
    if X.dim() == 1:
        X = X[None, :]
    device = X.device
    dtype = X.dtype

    x = l2_normalize_torch(X, dim=1) if normalize_inputs else X
    B, d = x.shape
    S = len(centers_all)

    for s, Cs in enumerate(centers_all):
        if Cs.dim() != 2 or Cs.size(1) != d:
            raise ValueError(f"Stage {s}: expected (K_s, {d}), got {tuple(Cs.shape)}")
        if Cs.device != device or Cs.dtype != dtype:
            raise ValueError(f"Stage {s}: device/dtype mismatch (Cs: {Cs.device}/{Cs.dtype}, X: {device}/{dtype})")

    cum = torch.zeros_like(x)                         # (B, d)
    codes = torch.empty(B, S, dtype=torch.long, device=device)
    cum_cos = torch.empty(B, S, dtype=dtype, device=device)

    for s, Cs in enumerate(centers_all):
        resid = l2_normalize_torch(x - cum, dim=1)    # (B, d)
        sims = resid @ Cs.t()                         # (B, K_s)
        idx = sims.argmax(dim=1)                      # (B,)
        codes[:, s] = idx
        chosen = Cs[idx, :]                           # (B, d)
        cum = cum + chosen
        cos_s = (x * l2_normalize_torch(cum, dim=1)).sum(dim=1)  # (B,)
        cum_cos[:, s] = cos_s

    return {"codes": codes, "cum_cos": cum_cos}


@torch.no_grad()
def assign_codes_in_chunks(
    X: torch.Tensor,
    centers_all: List[torch.Tensor],
    chunk_size: int = 65536,
    normalize_inputs: bool = True,
) -> Dict[str, torch.Tensor]:
    """Chunked version to keep memory under control on GPU."""
    B = X.size(0)
    codes_chunks = []
    cos_chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        out = assign_codes_batchwise(X[start:end], centers_all, normalize_inputs=normalize_inputs)
        codes_chunks.append(out["codes"])
        cos_chunks.append(out["cum_cos"])
    return {"codes": torch.cat(codes_chunks, dim=0), "cum_cos": torch.cat(cos_chunks, dim=0)}



def process_all(
    sts_df: pd.DataFrame,
    X: torch.Tensor,                       # (N, d) embeddings aligned to sts_df rows
    codebooks_root: str,
    device: torch.device,
    chunk_size: int,
    default_num_stages: int = 3,
) -> pd.DataFrame:
    """
    For each (topic, sentiment) group present in the STS dataframe:
      - find matching 'topic__sentiment' folder under codebooks_root
      - load its codebooks and batch-assign codes
    For groups with no folder, fill stage codes with -100 and cosine with NaN.
    Returns a dataframe with:
      ['global_index','stage1','stage_0','stage_1','stage_2','cosine_vq','topic','sentiment']
    """
    N = len(sts_df)
    if X.size(0) != N:
        raise ValueError(f"Embeddings rows ({X.size(0)}) != STS rows ({N}). They must align 1-to-1.")

    # Build mapping of available codebooks on disk
    avail = scan_codebooks_root(codebooks_root)  # {(topic, sentiment): Path}
    print(f"[info] Found {len(avail)} codebook folders under: {codebooks_root}")

    # Prepare output buffers
    # We'll determine S per group. To make a single table, we assume a common S for most groups.
    # If no codebooks exist at all, fall back to default_num_stages.
    sample_S: Optional[int] = None
    if len(avail) > 0:
        # peek one folder to infer S
        any_folder = next(iter(avail.values()))
        pt_path = any_folder / "codebooks_stage_all.pt"
        if pt_path.exists():
            S = len(load_codebooks_pt(str(pt_path)))
            sample_S = S
    if sample_S is None:
        sample_S = default_num_stages
    print(f"[info] Using sample number of stages: {sample_S}")

    # Initialize outputs with missing defaults (-100 and NaN)
    stage_cols = [f"stage_{s}" for s in range(sample_S)]
    out_codes = {col: np.full(N, -100, dtype=np.int64) for col in stage_cols}
    # 'stage1' is kept for backward compatibility (duplicates stage_0)
    out_codes["stage1"] = np.full(N, -100, dtype=np.int64)
    out_cos = np.full(N, np.nan, dtype=np.float32)

    # Group rows by (topic, sentiment)
    key_topic = sts_df["topic"]
    key_sent = sts_df["sentiment"]

    groups = {}
    for idx, (t, s) in enumerate(zip(key_topic.values, key_sent.values)):
        groups.setdefault((t, s), []).append(idx)
    print(f"[info] Found {len(groups)} unique (topic, sentiment) groups in the STS dataframe.")
    print(f"  Examples: {list(groups.keys())[:5]}")

    codebook_cache: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    for (t, s), row_indices in tqdm(groups.items(), desc="Processing groups", unit="group"):
        folder = avail.get((t, s), None)
        if folder is None:
            continue

        pt_codebooks = folder / "codebooks_stage_all.pt"
        if not pt_codebooks.exists():
            continue

        print(f"[info] Processing group (topic='{t}', sentiment='{s}') with {len(row_indices)} rows using codebooks from: {pt_codebooks}")

        if (t, s) not in codebook_cache:
            Cs_np = load_codebooks_pt(str(pt_codebooks))                 # list of (K_s, d), L2-normalized
            Cs_t = to_torch_codebooks(Cs_np, device=device)              # to torch tensors on device/dtype
            codebook_cache[(t, s)] = Cs_t

        Cs = codebook_cache[(t, s)]
        S_here = len(Cs)

        idxs = torch.as_tensor(row_indices, device=device, dtype=torch.long)
        X_sub = X.index_select(dim=0, index=idxs)  # (B, d)

        out = assign_codes_in_chunks(X_sub, Cs, chunk_size=chunk_size, normalize_inputs=True)
        codes_sub: torch.Tensor = out["codes"]     # (B, S_here)
        cos_sub: torch.Tensor = out["cum_cos"]     # (B, S_here)

        codes_np = codes_sub.detach().cpu().numpy()
        cos_np = cos_sub[:, -1].detach().cpu().numpy()  # final cumulative cosine

        for s_idx in range(S_here):
            col = f"stage_{s_idx}"
            if col not in out_codes:
                # expand columns if we discover more stages than sample_S
                out_codes[col] = np.full(N, -100, dtype=np.int64)
                stage_cols.append(col)
            out_codes[col][row_indices] = codes_np[:, s_idx]

        # stage1 (compat) mirrors stage_0 if available
        if "stage_0" in out_codes:
            out_codes["stage1"][row_indices] = codes_np[:, 0]

        out_cos[row_indices] = cos_np

        print(f"  Assigned {S_here} stages; example codes: {codes_np[0]}, cosine_vq: {cos_np[0]:.4f}")
        print()

    result = pd.DataFrame({
        "global_index": sts_df.index.astype(int),
        **{col: out_codes[col] for col in (["stage1"] + sorted([c for c in stage_cols]))},
        "cosine_vq": out_cos,
        "topic": sts_df["topic"].astype(str),
        "sentiment": sts_df["sentiment"].astype(str),
    })

    # Reorder columns: global_index, stage1, stage_0..stage_{S-1}, cosine_vq, topic, sentiment
    ordered_cols = ["global_index", "stage1"] + [f"stage_{s}" for s in range(len([c for c in stage_cols if c.startswith('stage_')]))] + ["cosine_vq", "topic", "sentiment"]
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    result = result[ordered_cols]

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical RVQ assignment by (topic, sentiment)."
    )
    parser.add_argument(
        "--sts_path", type=str, required=True,
        help="Path to the STS dataframe (CSV) with at least columns: statement, topic, sentiment."
    )
    parser.add_argument(
        "--embeddings_pt", type=str, required=True,
        help="Path to a .pt PyTorch tensor of shape (N, d) — must align with STS rows."
    )
    parser.add_argument(
        "--codebooks_root", type=str, required=True,
        help="Root folder containing subfolders named 'topic__sentiment', each with codebooks_stage_all.pt."
    )
    parser.add_argument(
        "--output_csv", type=str, default="vq_per_point_all.csv",
        help="Where to save the merged assignments CSV."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use, e.g., 'cuda', 'cuda:0', or 'cpu'. Default: auto (cuda if available)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=65536,
        help="Batch chunk size for assignment to limit GPU memory."
    )
    parser.add_argument(
        "--default_num_stages", type=int, default=3,
        help="Fallback number of stages if no codebooks are found at all."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[info] device={args.device}")

    sts_df = read_sts_dataframe(args.sts_path)
    print(f"[info] Loaded STS dataframe with {len(sts_df)} rows from: {args.sts_path}")
    print(sts_df.head())

    X = load_embeddings_pt(args.embeddings_pt, device=args.device)
    print(f"[info] Loaded embeddings tensor with shape {tuple(X.shape)} from: {args.embeddings_pt}")

    result_df = process_all(
        sts_df=sts_df,
        X=X,
        codebooks_root=args.codebooks_root,
        device=args.device,
        chunk_size=args.chunk_size,
        default_num_stages=args.default_num_stages,
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"[ok] Saved assignments to: {out_path}")


if __name__ == "__main__":
    main()