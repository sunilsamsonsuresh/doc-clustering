from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from transformers import LayoutLMv3Processor, LayoutLMv3Model

from .config import Config


def resolve_device(cfg: Config) -> str:
    if getattr(cfg, "device", "auto") in ("cpu", "cuda", "mps"):
        d = cfg.device
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


@torch.inference_mode()
def embed_layoutlmv3(
    cfg: Config,
    image_paths: list[Path],
    model_name: str = "microsoft/layoutlmv3-base",
) -> np.ndarray:
    """
    Creates document-level embeddings using LayoutLMv3 by:
      - Running OCR (via processor apply_ocr=True -> requires Tesseract)
      - Feeding image + tokens + bounding boxes into LayoutLMv3
      - Using CLS token embedding as the document embedding
    """
    device = resolve_device(cfg)
    # batch size: keep small; OCR is the slow part
    bs = 2 if device in ("cpu", "mps") else 4

    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=True)
    model = LayoutLMv3Model.from_pretrained(model_name).to(device)
    model.eval()

    X_parts = []
    for i in tqdm(range(0, len(image_paths), bs), desc="LayoutLMv3 embedding (OCR+layout)"):
        batch_paths = image_paths[i : i + bs]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        # Processor runs OCR and produces:
        # input_ids, attention_mask, bbox, pixel_values
        enc = processor(images=imgs, return_tensors="pt", truncation=True, padding=True)

        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        # out.last_hidden_state shape: (B, seq_len, hidden)
        # CLS token is first token
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden)
        X_parts.append(cls.detach().cpu().float().numpy())

    return np.vstack(X_parts)


def cluster_layoutlmv3(cfg: Config):
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.meta_csv)
    if "page1_png" not in df.columns:
        raise RuntimeError("meta.csv must contain 'page1_png' column for this pipeline.")

    image_paths = [Path(p) for p in df["page1_png"].tolist()]

    # 1) Embeddings
    model_name = getattr(cfg, "layoutlmv3_model", "microsoft/layoutlmv3-base")
    X = embed_layoutlmv3(cfg, image_paths, model_name=model_name)
    np.save(cfg.outputs_dir / "embeddings_layoutlmv3.npy", X)

    # 2) UMAP
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    Z = reducer.fit_transform(X)
    np.save(cfg.outputs_dir / "umap_layoutlmv3.npy", Z)

    # 3) HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Z)

    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out.to_csv(cfg.clusters_csv, index=False)

    # Evaluation (only if you have doc_type)
    mask = labels != -1
    ari = adjusted_rand_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"]) if mask.any() else 0.0
    nmi = normalized_mutual_info_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"]) if mask.any() else 0.0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = float(np.mean(labels == -1))
    print(f"clusters={n_clusters}  noise={noise:.2%}  ARI={ari:.3f}  NMI={nmi:.3f}")
    print(f"✅ wrote {cfg.clusters_csv}")


if __name__ == "__main__":
    cluster_layoutlmv3(Config())
