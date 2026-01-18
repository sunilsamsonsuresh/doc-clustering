from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm

from .config import Config

def render_first_page(pdf_path: Path, out_path: Path, zoom: float = 2.0):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out_path)
    doc.close()

def render_dataset(cfg: Config):
    df = pd.read_csv(cfg.meta_csv)
    cfg.page1_img_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Rendering page1"):
        pdf = Path(row.pdf_path)
        png = cfg.page1_img_dir / (pdf.stem + ".png")
        if not png.exists():
            render_first_page(pdf, png, zoom=2.0)

    print(f"✅ Rendered page-1 PNGs to: {cfg.page1_img_dir}")

if __name__ == "__main__":
    render_dataset(Config())
