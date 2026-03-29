"""
analytics.py — AgriVision Decision Support System
===================================================
Statistical analysis engine for field-level wheat head count data.
Provides Coefficient of Variation, yield estimation, and field health
classification from raw detection counts.


This module is strictly encapsulated agronomic computation.
"""

import numpy as np


class FieldAnalytics:
    """
    Encapsulates statistical analysis for a batch of per-image wheat head counts.

    Parameters
    ----------
    image_counts : list[int]
        Number of wheat heads detected in each image of the batch.
        Example: [42, 37, 51, 44] → four images, one count each.
    """

    def __init__(self, image_counts: list[int]):
        # Store as a NumPy array for vectorized computation
        self.image_counts = np.array(image_counts, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Coefficient of Variation
    # ------------------------------------------------------------------
    def calculate_cv(self) -> float:
        """
        Calculate the Coefficient of Variation (CV) of the image count batch.

        CV measures the relative spread of detections across images.
        A low CV indicates a spatially uniform field; a high CV reveals
        patchiness, localised stress, or irregular crop density.

        Formula:
            CV (%) = (std_dev / mean) × 100

        Returns
        -------
        float
            CV expressed as a percentage. Returns 0.0 if mean is zero
            to prevent ZeroDivisionError on empty or blank-field batches.
        """
        mean = float(np.mean(self.image_counts))
        if mean == 0:
            return 0.0
        std = float(np.std(self.image_counts, ddof=0))  # population std
        return (std / mean) * 100.0

    # ------------------------------------------------------------------
    # 2. Yield Estimation
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_yield(
        mean_count: float,
        tgw: float,
        grains_per_head: int,
        area_m2: float = 0.5,
    ) -> float:
        """
        Estimate wheat yield in tonnes per hectare (t/ha).

        Agronomic formula:
            yield_t_ha = (mean_count / area_m2) × grains_per_head
                         × (tgw / 1000) × 10000 / 1000

        Step-by-step breakdown:
            1. mean_count / area_m2          → heads per m²
            2. × grains_per_head             → grains per m²
            3. × (tgw / 1000)               → grams per m²  (TGW is per 1000 grains)
            4. × 10000                       → grams per hectare
            5. / 1000                        → kilograms → tonnes per hectare

        Parameters
        ----------
        mean_count : float
            Average wheat head count per image (equivalent to one field tile).
        tgw : float
            Thousand Grain Weight in grams (regional agronomic constant).
        grains_per_head : int
            Average number of grains per wheat head (regional agronomic constant).
        area_m2 : float, optional
            Ground area represented by one image tile in m².
            Default is 0.5 m² — calibrated for typical drone nadir imagery
            at 2–3 m altitude with a standard 1024×1024 crop window.

        Returns
        -------
        float
            Estimated yield in tonnes per hectare (t/ha).
        """
        heads_per_m2   = mean_count / area_m2
        grains_per_m2  = heads_per_m2 * grains_per_head
        grams_per_m2   = grains_per_m2 * (tgw / 1000.0)
        grams_per_ha   = grams_per_m2 * 10_000
        tonnes_per_ha  = grams_per_ha / 1_000_000
        return round(tonnes_per_ha, 4)

    # ------------------------------------------------------------------
    # 3. Field Health Classification
    # ------------------------------------------------------------------
    @staticmethod
    def get_health_status(cv: float) -> dict[str, str]:
        """
        Classify field health from the Coefficient of Variation.

        CV thresholds are based on agronomic precision-agriculture standards:
            < 15%  → Uniform crop stand (optimal)
            15–30% → Moderate spatial variance (watch zone)
            > 30%  → High patchiness / stress indicator (action required)

        Parameters
        ----------
        cv : float
            Coefficient of Variation percentage from calculate_cv().

        Returns
        -------
        dict with keys:
            status_color : str  — "Green" | "Yellow" | "Red"
            message      : str  — Human-readable agronomic interpretation.
        """
        if cv < 15.0:
            return {
                "status_color": "Green",
                "message"     : "Optimal: Field growth is highly uniform.",
            }
        elif cv <= 30.0:
            return {
                "status_color": "Yellow",
                "message"     : (
                    "Attention: Moderate variance detected. "
                    "Check for localized stress."
                ),
            }
        else:
            return {
                "status_color": "Red",
                "message"     : (
                    "Critical: High spatial variance. "
                    "Immediate field scout recommended."
                ),
            }
