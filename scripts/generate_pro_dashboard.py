"""
Generate Pro Dashboard — Iteration 2 Error Distribution Analysis
=================================================================
Reads outputs/iteration_2_tuned/eval_metrics/error_metadata.csv and produces
a research-grade 2×2 dashboard saved as pro_error_synthesis.png in the same
directory.

Charts:
  [0,0]  FP vs FN Distribution  — KDE / histogram overlay
  [0,1]  GT Density vs FN Count — scatter, size = Avg_Confidence
  [1,0]  Confidence Violin       — images with FP > 0 vs FP == 0
  [1,1]  Error Efficiency Index  — custom analytical insight
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
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
ROOT       = Path(__file__).resolve().parent.parent
EVAL_DIR   = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics"
CSV_PATH   = EVAL_DIR / "error_metadata.csv"
OUTPUT_PNG = EVAL_DIR / "pro_error_synthesis.png"


def load_data() -> pd.DataFrame:
    logger.info(f"📂 Loading evaluation metadata from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"   ✅ {len(df)} rows loaded | Columns: {list(df.columns)}")
    except FileNotFoundError:
        logger.error(f"❌ CSV not found: {CSV_PATH}")
        logger.error("   Run evaluate_model.py first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    # Derived columns
    df["Has_FP"]  = df["False_Positives"] > 0     # bool flag for violin split
    df["FP_Flag"] = df["Has_FP"].map({True: "Has FP (Hallucination)", False: "No FP (Clean)"})

    # Error Efficiency Index = TP / (TP + FP + FN)  — Jaccard Index per image
    denom = df["True_Positives"] + df["False_Positives"] + df["False_Negatives"]
    df["Jaccard"] = np.where(denom > 0, df["True_Positives"] / denom, 1.0)

    return df


def build_dashboard(df: pd.DataFrame):
    logger.info("🎨 Building 2×2 analytical dashboard...")

    # ── Global style ───────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid", palette="deep", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "#111827",
        "axes.facecolor":   "#1F2937",
        "axes.edgecolor":   "#374151",
        "axes.labelcolor":  "#D1D5DB",
        "xtick.color":      "#9CA3AF",
        "ytick.color":      "#9CA3AF",
        "grid.color":       "#374151",
        "text.color":       "#F9FAFB",
        "legend.facecolor": "#1F2937",
        "legend.edgecolor": "#374151",
    })

    C_FP  = "#F87171"   # Soft red     — hallucinations
    C_FN  = "#34D399"   # Emerald      — misses
    C_HAS = "#FB923C"   # Amber        — images with FP
    C_CLN = "#60A5FA"   # Sky blue     — clean images

    fig = plt.figure(figsize=(18, 13), facecolor="#111827")
    fig.suptitle(
        "Iteration 2: Error Distribution Analysis",
        fontsize=20, fontweight="bold", color="#F9FAFB", y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Chart 1: FP vs FN Distribution ────────────────────────────────────────
    logger.info("   [1/4] FP vs FN Distribution (KDE / Histogram overlay)...")
    sns.histplot(df["False_Positives"], kde=True, color=C_FP, alpha=0.5,
                 label="False Positives (Hallucinations)", ax=ax1, bins=30, stat="density")
    sns.histplot(df["False_Negatives"], kde=True, color=C_FN, alpha=0.5,
                 label="False Negatives (Misses)", ax=ax1, bins=30, stat="density")
    ax1.set_title("FP vs FN Distribution", fontsize=13, fontweight="bold", color="#F9FAFB", pad=8)
    ax1.set_xlabel("Count per Image")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)
    # Annotation: show means
    ax1.axvline(df["False_Positives"].mean(), color=C_FP, linestyle="--", linewidth=1.3,
                label=f"FP mean = {df['False_Positives'].mean():.1f}")
    ax1.axvline(df["False_Negatives"].mean(), color=C_FN, linestyle="--", linewidth=1.3,
                label=f"FN mean = {df['False_Negatives'].mean():.1f}")
    ax1.legend(fontsize=8.5)

    # ── Chart 2: GT Density vs FN Scatter ────────────────────────────────────
    logger.info("   [2/4] GT Density vs FN Count scatter...")
    # Scale marker sizes: small conf → small dot, high conf → big dot
    sizes = (df["Avg_Confidence"] * 120).clip(lower=8)
    scatter = ax2.scatter(
        df["Total_Ground_Truth"], df["False_Negatives"],
        c=df["Avg_Confidence"], cmap="plasma",
        s=sizes, alpha=0.65, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label("Avg Confidence", color="#D1D5DB", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#9CA3AF")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#9CA3AF")
    # Trend line
    m, b = np.polyfit(df["Total_Ground_Truth"], df["False_Negatives"], 1)
    xs = np.linspace(df["Total_Ground_Truth"].min(), df["Total_Ground_Truth"].max(), 100)
    ax2.plot(xs, m * xs + b, color="#FBBF24", linewidth=1.5, linestyle="--",
             label=f"Trend (slope={m:.3f})")
    ax2.set_title("GT Density vs False Negatives\n(size & color = Avg Confidence)",
                  fontsize=12, fontweight="bold", color="#F9FAFB", pad=8)
    ax2.set_xlabel("Total Ground Truth Boxes")
    ax2.set_ylabel("False Negatives")
    ax2.legend(fontsize=8.5)

    # ── Chart 3: Confidence Violin — FP vs Clean Images ───────────────────────
    logger.info("   [3/4] Confidence violin (FP vs Clean)...")
    sns.violinplot(
        data=df, x="FP_Flag", y="Avg_Confidence",
        hue="FP_Flag",
        palette={"Has FP (Hallucination)": C_HAS, "No FP (Clean)": C_CLN},
        ax=ax3, inner="quartile", linewidth=0.8, cut=0,
        order=["No FP (Clean)", "Has FP (Hallucination)"],
        legend=False,
    )
    ax3.set_title("Avg Confidence: Clean vs Hallucinating Images",
                  fontsize=12, fontweight="bold", color="#F9FAFB", pad=8)
    ax3.set_xlabel("Image Category")
    ax3.set_ylabel("Average Prediction Confidence")
    # Annotate medians
    for i, group in enumerate(["No FP (Clean)", "Has FP (Hallucination)"]):
        med = df.loc[df["FP_Flag"] == group, "Avg_Confidence"].median()
        ax3.text(i, med + 0.01, f"Md={med:.3f}", ha="center", fontsize=8.5,
                 color="#F9FAFB", fontweight="bold")

    # ── Chart 4: Custom — Jaccard Index Distribution by FP Bucket ─────────────
    # Insight: The Jaccard Index (IoU per image = TP / (TP+FP+FN)) reveals which
    # images suffer the most from combined localisation + detection failure.
    # We bucket FP count to expose the non-linear relationship between FP count
    # and spatial accuracy degradation.
    logger.info("   [4/4] Custom — Jaccard Index vs FP Severity buckets...")
    # Bins: [0,1) = zero FP, [1,3) = 1-2, [3,6) = 3-5, [6,11) = 6-10, [11,∞) = >10
    max_fp = int(df["False_Positives"].max()) + 2
    BINS   = [0, 1, 3, 6, 11, max_fp]
    LABELS = ["0 FP\n(Perfect)", "1–2 FP\n(Mild)", "3–5 FP\n(Moderate)",
              "6–10 FP\n(Severe)", ">10 FP\n(Critical)"]
    df["FP_Bucket"] = pd.cut(df["False_Positives"], bins=BINS, labels=LABELS,
                              right=False, include_lowest=True)

    palette_jac = dict(zip(LABELS, ["#34D399", "#A3E635", "#FBBF24", "#F87171", "#EF4444"]))
    sns.violinplot(
        data=df, x="FP_Bucket", y="Jaccard",
        hue="FP_Bucket", palette=palette_jac,
        ax=ax4, inner="quartile", linewidth=0.8, cut=0,
        order=LABELS, legend=False,
    )
    ax4.set_title("Jaccard Index (Spatial Accuracy) vs FP Severity\n"
                  "[Deep Insight: Does more FP → worse localisation?]",
                  fontsize=11, fontweight="bold", color="#F9FAFB", pad=8)
    ax4.set_xlabel("False Positive Severity Bucket")
    ax4.set_ylabel("Jaccard Index (TP / TP+FP+FN)")
    ax4.set_ylim(-0.05, 1.05)
    ax4.axhline(y=0.5, color="#9CA3AF", linestyle="--", linewidth=1.2, alpha=0.6,
                label="Jaccard = 0.50")
    ax4.legend(fontsize=8.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Saving dashboard at 300 DPI to: {OUTPUT_PNG}")
    try:
        fig.savefig(str(OUTPUT_PNG), dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        size_kb = OUTPUT_PNG.stat().st_size // 1024
        logger.info(f"✅ pro_error_synthesis.png saved ({size_kb:,} KB)")
    except Exception as e:
        logger.error(f"❌ Failed to save figure: {e}")
        sys.exit(1)


def main():
    logger.info("=" * 64)
    logger.info("  📊 Iteration 2 — Pro Error Synthesis Dashboard Generator")
    logger.info("=" * 64)
    df = load_data()
    build_dashboard(df)
    logger.info("=" * 64)
    logger.info(f"✅  All done. Dashboard: {OUTPUT_PNG}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
