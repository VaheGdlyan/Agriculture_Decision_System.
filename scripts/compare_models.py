"""
Baseline vs. Tuned — Model Comparison Showdown
===============================================
Reconstructs per-image TP / FP / FN stats for the Baseline model from its
stored JSON coordinate files, then loads the Iteration 2 eval_metrics CSV
(already per-image) and produces a research-grade 1×3 comparison dashboard.

Output: outputs/iteration_2_tuned/eval_metrics/model_comparison_showdown.png
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
ROOT = Path(__file__).resolve().parent.parent

# Baseline — reconstruct from JSON coordinates
BASELINE_FN_JSON = ROOT / "outputs" / "yolov8_baseline_640" / "fn_coordinates.json"
BASELINE_FP_JSON = ROOT / "outputs" / "yolov8_baseline_640" / "fp_coordinates.json"

# Iteration 2 — already a structured per-image CSV
ITER2_CSV = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics" / "error_metadata.csv"

# Ground truth label directory (to compute GT box counts per image)
LABELS_DIR = ROOT / "data" / "processed" / "yolo" / "labels" / "val"

OUTPUT_DIR = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics"
OUTPUT_PNG = OUTPUT_DIR / "model_comparison_showdown.png"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BASE  = "#E63946"   # Red  — Baseline (640 px)
C_TUNED = "#2A9D8F"   # Teal — Tuned (1024 px)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def count_gt_boxes(labels_dir: Path) -> dict[str, int]:
    """Return {image_stem: gt_box_count} from YOLO .txt label files."""
    counts = {}
    if labels_dir.exists():
        for f in labels_dir.glob("*.txt"):
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            counts[f.stem] = len(lines)
    return counts


def reconstruct_baseline(fn_json: Path, fp_json: Path,
                          gt_counts: dict) -> pd.DataFrame:
    """
    Build a per-image DataFrame matching the Iter2 CSV schema:
    Image_Name | Total_Ground_Truth | True_Positives | False_Positives |
    False_Negatives | Avg_Confidence
    """
    try:
        fn_data = json.loads(fn_json.read_text())
        fp_data = json.loads(fp_json.read_text())
    except FileNotFoundError as e:
        logger.error(f"❌ Baseline JSON not found: {e}")
        sys.exit(1)

    # Collect all image stems seen across both files
    all_stems = set()
    for k in fn_data:
        all_stems.add(Path(k).stem)
    for k in fp_data:
        all_stems.add(Path(k).stem)
    # Also include GT images with zero error entries
    all_stems.update(gt_counts.keys())

    # Build lookup by stem
    fn_by_stem: dict[str, int] = {}
    for k, v in fn_data.items():
        fn_by_stem[Path(k).stem] = len(v)

    fp_by_stem: dict[str, float] = {}    # stem → avg_conf
    fp_count_by_stem: dict[str, int] = {}
    for k, v in fp_data.items():
        stem = Path(k).stem
        fp_count_by_stem[stem] = len(v)
        confs = [r.get("conf", 0.0) if isinstance(r, dict) else 0.0 for r in v]
        fp_by_stem[stem] = float(np.mean(confs)) if confs else 0.0

    rows = []
    for stem in all_stems:
        gt  = gt_counts.get(stem, 0)
        fn  = fn_by_stem.get(stem, 0)
        fp  = fp_count_by_stem.get(stem, 0)
        tp  = max(0, gt - fn)
        avg_conf = fp_by_stem.get(stem, 0.0)
        rows.append({
            "Image_Name"        : stem + ".png",
            "Total_Ground_Truth": gt,
            "True_Positives"    : tp,
            "False_Positives"   : fp,
            "False_Negatives"   : fn,
            "Avg_Confidence"    : avg_conf,
        })

    df = pd.DataFrame(rows)
    logger.info(f"   Baseline reconstructed: {len(df):,} images")
    return df


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num.where(denom == 0, num / denom.replace(0, np.nan)).fillna(0.0)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Total_Ground_Truth"] > 0].copy()
    tp, fp, fn = df["True_Positives"], df["False_Positives"], df["False_Negatives"]
    df["Precision"] = safe_divide(tp, tp + fp)
    df["Recall"]    = safe_divide(tp, tp + fn)
    pr = df["Precision"] + df["Recall"]
    df["F1"] = safe_divide(2 * df["Precision"] * df["Recall"], pr)
    return df


# ─── Dashboard ────────────────────────────────────────────────────────────────
def build_dashboard(base_df: pd.DataFrame, tune_df: pd.DataFrame,
                    master: pd.DataFrame):
    logger.info("🎨 Building 1×3 comparison dashboard...")
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Baseline (640 px) vs. Tuned (1024 px) — Model Comparison Showdown",
        fontsize=15, fontweight="bold", y=1.02,
    )
    palette = {"Baseline (640px)": C_BASE, "Tuned (1024px)": C_TUNED}

    # ── Plot 1: Hallucination Cull — Violin ────────────────────────────────────
    logger.info("   [1/3] Hallucination Cull violin plot...")
    ax1 = axes[0]
    sns.violinplot(
        data=master, x="Model", y="False_Positives",
        hue="Model", palette=palette,
        order=["Baseline (640px)", "Tuned (1024px)"],
        ax=ax1, inner="quartile", cut=0, linewidth=0.8,
        legend=False,
    )
    ax1.set_title("The Hallucination Cull\n(Lower = Better)", fontweight="bold")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("False Positives per Image")
    # Median annotation
    for i, model in enumerate(["Baseline (640px)", "Tuned (1024px)"]):
        med = master.loc[master["Model"] == model, "False_Positives"].median()
        ax1.text(i, med + 0.3, f"Md={med:.1f}", ha="center", fontsize=9,
                 fontweight="bold",
                 color=C_BASE if "Baseline" in model else C_TUNED)

    # ── Plot 2: Confidence Density — KDE ──────────────────────────────────────
    logger.info("   [2/3] Confidence Density KDE...")
    ax2 = axes[1]
    for model, color in [("Baseline (640px)", C_BASE), ("Tuned (1024px)", C_TUNED)]:
        sub = master.loc[(master["Model"] == model) & (master["Avg_Confidence"] > 0),
                         "Avg_Confidence"]
        sns.kdeplot(sub, ax=ax2, color=color, fill=True, alpha=0.30,
                    linewidth=1.8, label=model)
        ax2.axvline(sub.median(), color=color, linestyle="--", linewidth=1.4,
                    label=f"{model} Md={sub.median():.3f}")
    ax2.set_title("Confidence Density\n(Tighter & Higher = Better)", fontweight="bold")
    ax2.set_xlabel("Average Prediction Confidence")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)

    # ── Plot 3: Aggregate System Performance — Grouped Bar ────────────────────
    logger.info("   [3/3] Aggregate performance bar chart...")
    ax3 = axes[2]
    metrics = ["Precision", "Recall", "F1"]
    x      = np.arange(len(metrics))
    width  = 0.32

    vals_base = [base_df[m].mean() for m in metrics]
    vals_tune = [tune_df[m].mean() for m in metrics]

    bars_b = ax3.bar(x - width / 2, vals_base, width, color=C_BASE,
                     label="Baseline (640px)", alpha=0.88)
    bars_t = ax3.bar(x + width / 2, vals_tune, width, color=C_TUNED,
                     label="Tuned (1024px)"   , alpha=0.88)

    for bars in (bars_b, bars_t):
        for bar in bars:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1.12)
    ax3.set_title("Aggregate System Performance\n(Higher = Better)", fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PNG), dpi=300, bbox_inches="tight")
    plt.close(fig)
    kb = OUTPUT_PNG.stat().st_size // 1024
    logger.info(f"✅ Saved: {OUTPUT_PNG.name} ({kb:,} KB)")


# ─── Terminal Summary Table ────────────────────────────────────────────────────
def print_summary(base_df: pd.DataFrame, tune_df: pd.DataFrame):
    def pct_change(old, new):
        return (new - old) / old * 100 if old else 0.0

    total_fp_b = base_df["False_Positives"].sum()
    total_fp_t = tune_df["False_Positives"].sum()
    total_fn_b = base_df["False_Negatives"].sum()
    total_fn_t = tune_df["False_Negatives"].sum()

    fp_delta = pct_change(total_fp_b, total_fp_t)
    fn_delta = pct_change(total_fn_b, total_fn_t)

    logger.info("")
    logger.info("┌──────────────────────────────────────────────────────────────┐")
    logger.info("│            MODEL COMPARISON SUMMARY — AGGREGATE              │")
    logger.info("├──────────────┬──────────────┬──────────────┬────────────────┤")
    logger.info("│ Metric       │ Baseline     │ Tuned        │ Δ Change        │")
    logger.info("├──────────────┼──────────────┼──────────────┼────────────────┤")
    logger.info(f"│ Total FP     │ {total_fp_b:<12,} │ {total_fp_t:<12,} │ {fp_delta:+.1f}%          │")
    logger.info(f"│ Total FN     │ {total_fn_b:<12,} │ {total_fn_t:<12,} │ {fn_delta:+.1f}%          │")
    logger.info(f"│ Precision    │ {base_df['Precision'].mean():.4f}        │ {tune_df['Precision'].mean():.4f}        │ {pct_change(base_df['Precision'].mean(), tune_df['Precision'].mean()):+.1f}%          │")
    logger.info(f"│ Recall       │ {base_df['Recall'].mean():.4f}        │ {tune_df['Recall'].mean():.4f}        │ {pct_change(base_df['Recall'].mean(), tune_df['Recall'].mean()):+.1f}%          │")
    logger.info(f"│ F1 Score     │ {base_df['F1'].mean():.4f}        │ {tune_df['F1'].mean():.4f}        │ {pct_change(base_df['F1'].mean(), tune_df['F1'].mean()):+.1f}%          │")
    logger.info("└──────────────┴──────────────┴──────────────┴────────────────┘")
    logger.info("")


# ─── Entry ────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 66)
    logger.info("  ⚔️  Baseline vs. Tuned — Model Comparison Pipeline")
    logger.info("=" * 66)

    # 1. GT box counts (shared val set)
    logger.info("📂 Counting GT boxes from label files...")
    gt_counts = count_gt_boxes(LABELS_DIR)
    logger.info(f"   {len(gt_counts):,} label files found")

    # 2. Reconstruct Baseline
    logger.info("🔧 Reconstructing Baseline per-image stats from JSON...")
    raw_base = reconstruct_baseline(BASELINE_FN_JSON, BASELINE_FP_JSON, gt_counts)

    # 3. Load Iteration 2
    logger.info("📂 Loading Iteration 2 eval_metrics CSV...")
    try:
        raw_tune = pd.read_csv(ITER2_CSV)
        logger.info(f"   Iteration 2 loaded: {len(raw_tune):,} rows")
    except FileNotFoundError:
        logger.error(f"❌ Iter2 CSV not found: {ITER2_CSV}")
        sys.exit(1)

    # 4. Tag models
    raw_base["Model"] = "Baseline (640px)"
    raw_tune["Model"] = "Tuned (1024px)"

    # 5. Feature engineering
    logger.info("⚙️  Computing Precision / Recall / F1 per image...")
    base_df = compute_metrics(raw_base)
    tune_df = compute_metrics(raw_tune)

    # 6. Master DataFrame
    master = pd.concat([base_df, tune_df], ignore_index=True)
    logger.info(f"   Master DataFrame: {len(master):,} rows after dropping GT=0 images")

    # 7. Terminal summary
    print_summary(base_df, tune_df)

    # 8. Dashboard
    build_dashboard(base_df, tune_df, master)

    logger.info("=" * 66)
    logger.info(f"✅  All done → {OUTPUT_PNG}")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
