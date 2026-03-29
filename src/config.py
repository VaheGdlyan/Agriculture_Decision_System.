"""
config.py — AgriVision Decision Support System
================================================
Global configuration module. Contains strictly encapsulated agronomic
constants used for yield estimation calculations.

DO NOT add UI, Streamlit, or inference logic here.
This file is the single source of truth for all regional baselines.

Agronomic Formula Reference:
  Yield (t/ha) = (wheat_head_count × grains_per_head × tgw_grams) / 1_000_000
  Revenue      = Yield × price_per_tonne_usd
"""

# ---------------------------------------------------------------------------
# REGIONAL_DEFAULTS
# ---------------------------------------------------------------------------
# A lookup table of agronomic baselines per region.
#
# Keys per region:
#   tgw_grams          — Thousand Grain Weight in grams. The average mass of
#                        1,000 wheat grains for that region's dominant variety.
#                        Typical range: 35–55 g. Heavier = higher yield.
#
#   grains_per_head    — Average number of grains per wheat spike/head.
#                        Determined by the regional variety and climate.
#                        Typical range: 30–55 grains.
#
#   price_per_tonne_usd — Estimated farm-gate wheat price in USD per tonne.
#                         Based on FAO & World Bank 2023–2024 benchmarks.
#                         Used for revenue estimation output.
#
#   currency           — ISO 4217 code for the local currency.
#                         Displayed in the UI alongside the USD estimate.
# ---------------------------------------------------------------------------

REGIONAL_DEFAULTS: dict[str, dict] = {

    "Armenia": {
        # Dominant variety: soft bread wheat (Harutyun, Nairi).
        # Short growing season in Ararat Valley; moderate TGW.
        "tgw_grams"          : 40.0,
        "grains_per_head"    : 38,
        "price_per_tonne_usd": 210.0,
        "currency"           : "AMD",
    },

    "United States": {
        # Dominant varieties: Hard Red Winter (Kansas), SRRS.
        # High mechanization, large-scale production.
        "tgw_grams"          : 46.0,
        "grains_per_head"    : 42,
        "price_per_tonne_usd": 245.0,
        "currency"           : "USD",
    },

    "Ukraine": {
        # Dominant variety: soft and durum winter wheat (Smuhlyanka, Podolyanka).
        # Chernozem (black earth) belt — historically one of highest-yielding regions.
        "tgw_grams"          : 44.0,
        "grains_per_head"    : 45,
        "price_per_tonne_usd": 225.0,
        "currency"           : "UAH",
    },

    "India": {
        # Dominant variety: HD-2967, PBW-343 (Punjab, Haryana, UP).
        # Short grain-fill period due to terminal heat stress.
        "tgw_grams"          : 38.0,
        "grains_per_head"    : 35,
        "price_per_tonne_usd": 195.0,
        "currency"           : "INR",
    },

    "China": {
        # Dominant variety: Yangmai, Zhongmai series.
        # Varies significantly across Yellow River plain vs. southwestern region.
        "tgw_grams"          : 42.0,
        "grains_per_head"    : 40,
        "price_per_tonne_usd": 230.0,
        "currency"           : "CNY",
    },

    "France": {
        # Dominant variety: Hybery, Rubisko (soft bread wheat).
        # High agronomy intensity, EU CAP-supported — consistently high TGW.
        "tgw_grams"          : 50.0,
        "grains_per_head"    : 44,
        "price_per_tonne_usd": 255.0,
        "currency"           : "EUR",
    },

    "Australia": {
        # Dominant variety: Mace, Scepter (APW/ASW grades).
        # Southern dryland wheat belt; TGW can vary with rainfall.
        "tgw_grams"          : 43.0,
        "grains_per_head"    : 38,
        "price_per_tonne_usd": 240.0,
        "currency"           : "AUD",
    },

    "Canada": {
        # Dominant variety: CWRS (Canada Western Red Spring).
        # Manitoba / Saskatchewan — premium high-protein varieties.
        "tgw_grams"          : 48.0,
        "grains_per_head"    : 40,
        "price_per_tonne_usd": 250.0,
        "currency"           : "CAD",
    },

    "Russia": {
        # Dominant variety: Saratovskaya, Bezenchukskaya (spring wheat, Volga).
        # World's largest exporter; broad climate range lowers average TGW.
        "tgw_grams"          : 41.0,
        "grains_per_head"    : 38,
        "price_per_tonne_usd": 215.0,
        "currency"           : "RUB",
    },

    "Argentina": {
        # Dominant variety: Klein Tauro, Baguette (Pampa region).
        # Southern hemisphere planting calendar (planted June, harvested Dec).
        "tgw_grams"          : 44.0,
        "grains_per_head"    : 42,
        "price_per_tonne_usd": 220.0,
        "currency"           : "ARS",
    },

    "Custom Setup": {
        # Fallback / user-defined region.
        # UI should allow the user to override all four values manually.
        # Defaults use global FAO average benchmarks.
        "tgw_grams"          : 42.0,
        "grains_per_head"    : 40,
        "price_per_tonne_usd": 220.0,
        "currency"           : "USD",
    },
}

# ---------------------------------------------------------------------------
# Convenience helpers (read-only)
# ---------------------------------------------------------------------------

SUPPORTED_REGIONS: list[str] = list(REGIONAL_DEFAULTS.keys())
"""Ordered list of all supported region names, for UI dropdown population."""

DEFAULT_REGION: str = "Custom Setup"
"""Fallback region key used when no region is selected by the user."""
