from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from transformers import AutoImageProcessor, AutoModel

from .config import Config


def resolve_device(cfg: Config) -> str:
    d = getattr(cfg, "device", "auto")
    if d in ("cpu", "cuda", "mps"):
        if d == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if d == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return d
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


@torch.inference_mode()
def embed_dit(
    cfg: Config,
    image_paths: list[Path],
    model_name: str = "microsoft/dit-base-finetuned-rvlcdip",
) -> np.ndarray:
    """
    Create image embeddings using DiT (Document Image Transformer).

    We try to extract a page-level embedding in this priority order:
      1) outputs.pooler_output  (if available)
      2) CLS token from last_hidden_state ([:,0,:])  (if available)
      3) mean pool over last_hidden_state tokens       (fallback)

    Notes:
      - We use AutoModel (not classifier head) to get hidden states.
      - Some DiT checkpoints return last_hidden_state directly.
    """
    device = resolve_device(cfg)

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # batch size (tuneable)
    if device == "cuda":
        bs = getattr(cfg, "batch_size", 32)
    elif device == "mps":
        bs = max(2, getattr(cfg, "batch_size", 16) // 4)
    else:
        bs = max(2, getattr(cfg, "batch_size", 16) // 8)

    X_parts: list[np.ndarray] = []

    use_amp = device in ("cuda", "mps")
    amp_ctx = torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp)

    for i in tqdm(range(0, len(image_paths), bs), desc="DiT embedding"):
        batch_paths = image_paths[i : i + bs]
        imgs = [load_image(p) for p in batch_paths]

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with amp_ctx:
            out = model(**inputs, output_hidden_states=False, return_dict=True)

            # Try pooler_output first
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output  # (B, hidden)

            # Try last_hidden_state (common for ViT-like)
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hs = out.last_hidden_state  # (B, seq, hidden)
                # CLS token if present
                emb = hs[:, 0, :] if hs.shape[1] >= 1 else hs.mean(dim=1)

            else:
                raise RuntimeError("Model output has no pooler_output or last_hidden_state; can't extract embeddings.")

        X_parts.append(emb.detach().cpu().float().numpy())

        if device == "mps":
            torch.mps.empty_cache()

    return np.vstack(X_parts)


def cluster_dit(cfg: Config):
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.meta_csv)

    # Prefer page1_path; fallback to page1_png
    if "page1_path" in df.columns:
        image_paths = [Path(p) for p in df["page1_path"].tolist()]
    elif "page1_png" in df.columns:
        image_paths = [Path(p) for p in df["page1_png"].tolist()]
    else:
        raise RuntimeError("meta.csv must contain 'page1_path' or 'page1_png' column.")

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing. Example: {missing[0]}")

    model_name = getattr(cfg, "dit_model", "microsoft/dit-base-finetuned-rvlcdip")

    # 1) Embeddings
    X = embed_dit(cfg, image_paths, model_name=model_name)
    np.save(cfg.outputs_dir / "embeddings_dit.npy", X)

    # 2) UMAP
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    Z = reducer.fit_transform(X)
    np.save(cfg.outputs_dir / "umap_dit.npy", Z)

    # 3) HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Z)

    df_out = df.copy()
    df_out["cluster_id"] = labels

    out_csv = cfg.outputs_dir / "clusters_dit.csv"
    df_out.to_csv(out_csv, index=False)

    # Optional evaluation if doc_type exists
    mask = labels != -1
    if mask.any() and "doc_type" in df_out.columns:
        ari = adjusted_rand_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"])
        nmi = normalized_mutual_info_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"])
    else:
        ari, nmi = 0.0, 0.0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = float(np.mean(labels == -1))

    print(f"✅ DiT clusters={n_clusters}  noise={noise:.2%}  ARI={ari:.3f}  NMI={nmi:.3f}")
    print(f"✅ wrote {out_csv}")


if __name__ == "__main__":
    cluster_dit(Config())
