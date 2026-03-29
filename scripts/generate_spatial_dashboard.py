"""
Spatial & Size Error Analysis Dashboard — Iteration 2
======================================================
Parses fn_coordinates.json and fp_coordinates.json to extract bounding box
geometry (center X, center Y, width, height, area) and produces a 1×3
spatial analysis dashboard.

Charts:
  [0]  Spatial Heatmap — FP (Hallucinations) — red KDE on image canvas
  [1]  Spatial Heatmap — FN (Misses)         — blue KDE on image canvas
  [2]  Area Distribution — FP vs FN          — histogram + KDE overlay

Output: outputs/iteration_2_tuned/eval_metrics/spatial_size_error_analysis.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Dynamic Paths ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
EVAL_DIR  = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics"
FN_JSON   = EVAL_DIR / "fn_coordinates.json"
FP_JSON   = EVAL_DIR / "fp_coordinates.json"
OUTPUT    = EVAL_DIR / "spatial_size_error_analysis.png"

# Normalisation reference — our images are 1024×1024
IMG_W = IMG_H = 1024


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_fn(path: Path) -> pd.DataFrame:
    """FN JSON: { img: [[x1,y1,x2,y2], ...] }"""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        logger.error(f"❌ Not found: {path}")
        sys.exit(1)

    rows = []
    for boxes in data.values():
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            rows.append({
                "cx"  : x1 + w / 2,
                "cy"  : y1 + h / 2,
                "w"   : w,
                "h"   : h,
                "area": w * h,
            })
    df = pd.DataFrame(rows)
    logger.info(f"   FN boxes loaded : {len(df):,}")
    return df


def load_fp(path: Path) -> pd.DataFrame:
    """FP JSON: { img: [{"box": [x1,y1,x2,y2], "conf": float}, ...] }"""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        logger.error(f"❌ Not found: {path}")
        sys.exit(1)

    rows = []
    for records in data.values():
        for rec in records:
            x1, y1, x2, y2 = rec["box"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            rows.append({
                "cx"  : x1 + w / 2,
                "cy"  : y1 + h / 2,
                "w"   : w,
                "h"   : h,
                "area": w * h,
                "conf": rec.get("conf", 0.0),
            })
    df = pd.DataFrame(rows)
    logger.info(f"   FP boxes loaded : {len(df):,}")
    return df


# ─── Dashboard ────────────────────────────────────────────────────────────────
def build_dashboard(fn: pd.DataFrame, fp: pd.DataFrame):
    logger.info("🎨 Building 1×3 spatial dashboard...")

    sns.set_theme(style="darkgrid", font_scale=1.05)
    BG   = "#0F172A"
    AX   = "#1E293B"
    GRID = "#334155"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor"  : AX,
        "axes.edgecolor"  : GRID,
        "axes.labelcolor" : "#CBD5E1",
        "xtick.color"     : "#94A3B8",
        "ytick.color"     : "#94A3B8",
        "grid.color"      : GRID,
        "text.color"      : "#F1F5F9",
        "legend.facecolor": AX,
        "legend.edgecolor": GRID,
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    fig.suptitle(
        "Iteration 2 — Spatial & Size Error Analysis",
        fontsize=17, fontweight="bold", color="#F1F5F9", y=1.02,
    )

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(AX)
        ax.set_title(title, fontsize=12, fontweight="bold", color="#F1F5F9", pad=10)
        ax.set_xlabel(xlabel, color="#CBD5E1")
        ax.set_ylabel(ylabel, color="#CBD5E1")

    # ── Chart 1: FP Spatial Heatmap ───────────────────────────────────────────
    logger.info("   [1/3] FP spatial KDE heatmap...")
    ax1 = axes[0]
    try:
        sns.kdeplot(
            data=fp, x="cx", y="cy",
            fill=True, cmap="Reds",
            levels=15, thresh=0.02,
            ax=ax1,
        )
    except Exception as e:
        logger.warning(f"      KDE failed for FP ({e}), using scatter fallback")
        ax1.scatter(fp["cx"], fp["cy"], c="#F87171", alpha=0.15, s=4)

    ax1.set_xlim(0, IMG_W)
    ax1.set_ylim(0, IMG_H)
    ax1.invert_yaxis()            # image origin is top-left
    ax1.set_aspect("equal")
    # Canvas border
    for spine in ax1.spines.values():
        spine.set_edgecolor("#EF4444")
        spine.set_linewidth(1.5)
    style(ax1,
          "Spatial Heatmap — Hallucinations (FP)\n"
          f"({len(fp):,} false positive boxes)",
          "Center X (px)", "Center Y (px)")
    ax1.text(0.02, 0.98, f"n = {len(fp):,}", transform=ax1.transAxes,
             fontsize=9, color="#FCA5A5", va="top")

    # ── Chart 2: FN Spatial Heatmap ───────────────────────────────────────────
    logger.info("   [2/3] FN spatial KDE heatmap...")
    ax2 = axes[1]
    try:
        sns.kdeplot(
            data=fn, x="cx", y="cy",
            fill=True, cmap="Blues",
            levels=15, thresh=0.02,
            ax=ax2,
        )
    except Exception as e:
        logger.warning(f"      KDE failed for FN ({e}), using scatter fallback")
        ax2.scatter(fn["cx"], fn["cy"], c="#60A5FA", alpha=0.15, s=4)

    ax2.set_xlim(0, IMG_W)
    ax2.set_ylim(0, IMG_H)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#3B82F6")
        spine.set_linewidth(1.5)
    style(ax2,
          "Spatial Heatmap — Misses (FN)\n"
          f"({len(fn):,} false negative boxes)",
          "Center X (px)", "Center Y (px)")
    ax2.text(0.02, 0.98, f"n = {len(fn):,}", transform=ax2.transAxes,
             fontsize=9, color="#93C5FD", va="top")

    # ── Chart 3: Area Distribution Overlay ────────────────────────────────────
    logger.info("   [3/3] Box area distribution comparison...")
    ax3 = axes[2]

    # Cap extreme outliers for readability (99th percentile)
    area_cap = max(fp["area"].quantile(0.99), fn["area"].quantile(0.99))
    fp_area  = fp["area"].clip(upper=area_cap)
    fn_area  = fn["area"].clip(upper=area_cap)

    sns.histplot(fp_area, kde=True, color="#F87171", alpha=0.45,
                 label=f"FP — Hallucinations (n={len(fp):,})",
                 stat="density", bins=35, ax=ax3)
    sns.histplot(fn_area, kde=True, color="#60A5FA", alpha=0.45,
                 label=f"FN — Misses (n={len(fn):,})",
                 stat="density", bins=35, ax=ax3)

    # Median markers
    for val, color, label in [
        (fp["area"].median(), "#EF4444", f"FP median={fp['area'].median():.0f}"),
        (fn["area"].median(), "#3B82F6", f"FN median={fn['area'].median():.0f}"),
    ]:
        ax3.axvline(val, color=color, linestyle="--", linewidth=1.5, label=label)

    ax3.set_title("Error by Object Size (Area)\n"
                  "Do failures cluster on small or large objects?",
                  fontsize=11, fontweight="bold", color="#F1F5F9", pad=10)
    ax3.set_xlabel("Bounding Box Area (px²)")
    ax3.set_ylabel("Density")
    ax3.set_facecolor(AX)
    ax3.legend(fontsize=8.5, loc="upper right")
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    kb = OUTPUT.stat().st_size // 1024
    logger.info(f"✅ Saved: {OUTPUT.name} ({kb:,} KB)")


# ─── Entry ────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 64)
    logger.info("  🗺️  Spatial & Size Error Analysis Dashboard")
    logger.info("=" * 64)
    logger.info("📂 Loading JSON coordinate files...")
    fn_df = load_fn(FN_JSON)
    fp_df = load_fp(FP_JSON)
    build_dashboard(fn_df, fp_df)
    logger.info("=" * 64)
    logger.info(f"✅  Dashboard: {OUTPUT}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
