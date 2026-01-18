from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from .config import Config


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_to_pdf(img_path: Path, out_pdf: Path):
    """Wrap a single image into a 1-page PDF so you can 'paste docs into cluster folders'."""
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4
    c.drawImage(
        str(img_path),
        15 * mm,
        20 * mm,
        width=w - 30 * mm,
        height=h - 40 * mm,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.showPage()
    c.save()


def prepare_local_rvl(cfg: Config):
    """
    Expects:
      data/<class_name>/*.(tif|png|jpg|...)
    Produces:
      data/page1_pngs/*.png
      data/raw_pdfs/*.pdf
      data/meta.csv
    """
    random.seed(cfg.seed)

    cfg.data_dir.mkdir(exist_ok=True)
    cfg.page1_img_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    root = cfg.data_dir

    # class folders are immediate subfolders under data/
    # SELECTED_FOLDERS = {
    # "invoice",
    # "form",
    # "email",
    # "handwritten",
    # "letter"
    # }

    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name not in {"raw_pdfs", "page1_pngs"}]
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {root}")

    samples = []
    for cls_dir in sorted(class_dirs):
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((cls_dir.name, p))

    if not samples:
        raise RuntimeError(f"No images found under {root}. Expected data/<class>/*.tif|png|jpg...")

    random.shuffle(samples)
    samples = samples[: cfg.n_docs]

    rows = []
    for i, (label, src_path) in tqdm(list(enumerate(samples)), desc="Preparing local dataset"):
        base = f"{i:04d}_{label}"
        png_path = cfg.page1_img_dir / f"{base}.png"
        pdf_path = cfg.raw_pdf_dir / f"{base}.pdf"

        # Normalize to PNG (DINOv2 input)
        if not png_path.exists():
            img = Image.open(src_path).convert("RGB")
            img.save(png_path)

        # Create PDF wrapper (for folder-per-cluster deliverable)
        if not pdf_path.exists():
            image_to_pdf(png_path, pdf_path)

        rows.append(
            {
                "doc_type": label,
                "pdf_path": str(pdf_path),
                "page1_png": str(png_path),
                "source_path": str(src_path),
            }
        )

    pd.DataFrame(rows).to_csv(cfg.meta_csv, index=False)
    print(f"✅ Prepared {len(rows)} items")
    print(f"✅ meta.csv -> {cfg.meta_csv}")
    print(f"✅ PNGs -> {cfg.page1_img_dir}")
    print(f"✅ PDFs -> {cfg.raw_pdf_dir}")


if __name__ == "__main__":
    prepare_local_rvl(Config())
