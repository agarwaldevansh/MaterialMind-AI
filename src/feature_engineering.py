"""
feature_engineering.py - MaterialMind AI
=========================================
Creates engineered features from cleaned steel datasets.

CRITICAL: No target leakage!
- Target columns are NEVER used as ML input features.
- yield_tensile_ratio is for DISPLAY/COMPARISON only, never for ML.
"""

import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------------
# Constants: define which columns are targets vs chemical elements
# ---------------------------------------------------------------------------

# Dataset 1 (steel_db) chemical element columns
STEEL_DB_ELEMENTS = [
    "c", "mn", "si", "cr", "ni", "mo", "cu", "v", "al", "ti",
    "s", "p", "fe", "co", "nb", "b", "n", "zr"
]

# Dataset 2 (matnavi) chemical element columns
MATNAVI_ELEMENTS = [
    "c", "si", "mn", "p", "s", "ni", "cr", "mo", "cu", "v", "al", "n"
]

# Target columns that must NEVER be used as ML input features
STEEL_DB_TARGETS = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
MATNAVI_TARGETS = ["tensile_strength_mpa", "proof_stress_mpa",
                   "elongation_pct", "reduction_in_area_pct"]

# Display-only columns (computed from targets — NOT for ML)
DISPLAY_ONLY_COLS = ["yield_tensile_ratio_DISPLAY_ONLY"]


# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------

def add_composition_features(df, dataset_name):
    """
    Add composition-based engineered features to the dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The cleaned dataset.
    dataset_name : str
        Either 'steel_db' or 'matnavi'.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with new engineered feature columns added.
    """

    if dataset_name == "steel_db":
        # ---- Feature 1: total_alloying_pct ----
        # Sum of exactly 17 alloying elements (excluding iron 'fe'):
        # ['al', 'b', 'c', 'co', 'cr', 'cu', 'mn', 'mo', 'n', 'nb', 'ni', 'p', 's', 'si', 'ti', 'v', 'zr']
        # Measures the overall alloying level (wt%).
        alloying_elements = [col for col in STEEL_DB_ELEMENTS if col != "fe"]
        available_alloying = [col for col in alloying_elements if col in df.columns]
        df["total_alloying_pct"] = df[available_alloying].sum(axis=1)
        print(f"  Added total_alloying_pct: sum of {len(available_alloying)} elements ({', '.join(sorted(available_alloying))})")

        # ---- Feature 2: carbon_equivalent (IIW formula) ----
        # Formula: C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15
        # Standard IIW (International Institute of Welding) formula.
        # Units: weight percent equivalent (wt%). Evaluates weldability/hardenability.
        df["carbon_equivalent"] = (
            df.get("c", 0)
            + df.get("mn", 0) / 6
            + (df.get("cr", 0) + df.get("mo", 0) + df.get("v", 0)) / 5
            + (df.get("ni", 0) + df.get("cu", 0)) / 15
        )
        print("  Added carbon_equivalent (IIW formula, wt% equivalent)")

        # ---- Feature 3: mn_c_ratio ----
        # Elemental composition ratio (Mn wt% / C wt%).
        # Note: Described strictly as a composition ratio (safe division: 0 if C = 0).
        df["mn_c_ratio"] = np.where(
            df.get("c", 0) == 0,
            0,
            df.get("mn", 0) / df["c"]
        )
        print("  Added mn_c_ratio (elemental composition ratio: Mn/C, 0 when C=0)")

        # ---- Feature 4: cr_mo_total ----
        # Simple additive composition feature: Cr wt% + Mo wt%.
        # Reflects the combined content of major carbide-forming elements in Cr-Mo steels (e.g. AISI 41xx).
        df["cr_mo_total"] = df.get("cr", 0) + df.get("mo", 0)
        print("  Added cr_mo_total (additive Cr + Mo wt% content)")

        # ---- Feature 5: ni_cr_total ----
        # Simple additive composition feature: Ni wt% + Cr wt%.
        # Reflects primary alloying additions in Ni-Cr and Ni-Cr-Mo steels (e.g. AISI 43xx, 86xx).
        df["ni_cr_total"] = df.get("ni", 0) + df.get("cr", 0)
        print("  Added ni_cr_total (additive Ni + Cr wt% content)")

    elif dataset_name == "matnavi":
        # ---- Feature 1: total_alloying_pct ----
        # Sum of 13 composition elements: ['c', 'si', 'mn', 'p', 's', 'ni', 'cr', 'mo', 'cu', 'v', 'al', 'n', 'nb_ta']
        sum_cols = MATNAVI_ELEMENTS + ["nb_ta"]
        available_sum = [col for col in sum_cols if col in df.columns]
        df["total_alloying_pct"] = df[available_sum].sum(axis=1)
        print(f"  Added total_alloying_pct: sum of {len(available_sum)} composition elements ({', '.join(sorted(available_sum))})")

        # ---- Feature 2: carbon_equivalent_calc (IIW formula) ----
        # Formula: C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15
        # Units: weight percent equivalent (wt%). Computed to provide consistent CE across records.
        df["carbon_equivalent_calc"] = (
            df.get("c", 0)
            + df.get("mn", 0) / 6
            + (df.get("cr", 0) + df.get("mo", 0) + df.get("v", 0)) / 5
            + (df.get("ni", 0) + df.get("cu", 0)) / 15
        )
        print("  Added carbon_equivalent_calc (IIW formula, wt% equivalent)")

        # ---- Feature 3: mn_c_ratio ----
        # Elemental composition ratio (Mn wt% / C wt%). Safe division: 0 if C = 0.
        df["mn_c_ratio"] = np.where(
            df.get("c", 0) == 0,
            0,
            df.get("mn", 0) / df["c"]
        )
        print("  Added mn_c_ratio (elemental composition ratio: Mn/C, 0 when C=0)")

        # ---- Feature 4: cr_mo_total ----
        # Simple additive composition feature: Cr wt% + Mo wt%.
        df["cr_mo_total"] = df.get("cr", 0) + df.get("mo", 0)
        print("  Added cr_mo_total (additive Cr + Mo wt% content)")

        # ---- Feature 5: is_elevated_temp (binary temperature indicator) ----
        # Binary flag: 1 if temperature > 100°C, else 0 (ambient/near-ambient vs elevated)
        if "temperature_c" in df.columns:
            df["is_elevated_temp"] = (df["temperature_c"] > 100).astype(int)
            print("  Added is_elevated_temp (binary flag: 1 if temperature_c > 100°C, else 0)")
        else:
            print("  WARNING: temperature_c column not found — skipping is_elevated_temp")

    else:
        print(f"  ERROR: Unknown dataset_name '{dataset_name}'. Use 'steel_db' or 'matnavi'.")

    return df


def add_display_features(df, dataset_name):
    """
    Add post-prediction display features that are NOT for ML input.

    These features use target columns (like tensile/yield strength) and
    are meant for analysis and comparison ONLY — never as model inputs.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset (must already contain target columns).
    dataset_name : str
        Either 'steel_db' or 'matnavi'.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with display-only feature columns added.
    """

    if dataset_name == "steel_db":
        # yield_strength / tensile_strength ratio — DISPLAY ONLY
        if "yield_strength_mpa" in df.columns and "tensile_strength_mpa" in df.columns:
            df["yield_tensile_ratio_DISPLAY_ONLY"] = np.where(
                df["tensile_strength_mpa"] == 0,
                0,
                df["yield_strength_mpa"] / df["tensile_strength_mpa"]
            )
            print("  Added yield_tensile_ratio_DISPLAY_ONLY (yield/tensile — NOT for ML!)")

    elif dataset_name == "matnavi":
        # proof_stress / tensile_strength ratio — DISPLAY ONLY
        if "proof_stress_mpa" in df.columns and "tensile_strength_mpa" in df.columns:
            df["yield_tensile_ratio_DISPLAY_ONLY"] = np.where(
                df["tensile_strength_mpa"] == 0,
                0,
                df["proof_stress_mpa"] / df["tensile_strength_mpa"]
            )
            print("  Added yield_tensile_ratio_DISPLAY_ONLY (proof_stress/tensile — NOT for ML!)")

    return df


def get_ml_feature_columns(dataset_name):
    """
    Return a list of column names that are safe to use as ML input features.

    IMPORTANT: This list excludes ALL target columns and display-only columns
    to prevent data leakage.

    Parameters
    ----------
    dataset_name : str
        Either 'steel_db' or 'matnavi'.

    Returns
    -------
    feature_cols : list of str
        Column names safe for ML input.
    """

    if dataset_name == "steel_db":
        # Original chemical element columns + engineered composition features
        feature_cols = (
            STEEL_DB_ELEMENTS
            + ["total_alloying_pct", "carbon_equivalent", "mn_c_ratio",
               "cr_mo_total", "ni_cr_total"]
        )

    elif dataset_name == "matnavi":
        # Original chemical elements + nb_ta + ceq + temperature + engineered features
        feature_cols = (
            MATNAVI_ELEMENTS
            + ["nb_ta", "ceq", "temperature_c",
               "total_alloying_pct", "carbon_equivalent_calc", "mn_c_ratio",
               "cr_mo_total", "is_elevated_temp"]
        )

    else:
        print(f"ERROR: Unknown dataset_name '{dataset_name}'.")
        return []

    # Safety check: make sure no target or display columns snuck in
    forbidden = set(STEEL_DB_TARGETS + MATNAVI_TARGETS + DISPLAY_ONLY_COLS)
    feature_cols = [col for col in feature_cols if col not in forbidden]

    return feature_cols


def get_target_columns(dataset_name):
    """
    Return a list of target column names for the given dataset.

    Parameters
    ----------
    dataset_name : str
        Either 'steel_db' or 'matnavi'.

    Returns
    -------
    target_cols : list of str
        Target column names.
    """

    if dataset_name == "steel_db":
        return STEEL_DB_TARGETS.copy()
    elif dataset_name == "matnavi":
        return MATNAVI_TARGETS.copy()
    else:
        print(f"ERROR: Unknown dataset_name '{dataset_name}'.")
        return []


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_feature_engineering():
    """
    Main function: load cleaned data, add engineered features, and save
    the results to the data/processed/ folder.

    Outputs:
        - data/processed/steel_db_features.csv
        - data/processed/matnavi_features.csv
    """

    # Set up file paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_dir, "data", "processed")
    cleaned_dir = processed_dir

    # Create the processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset 1: steel_db
    # ------------------------------------------------------------------
    steel_db_path = os.path.join(cleaned_dir, "steel_db_cleaned.csv")
    if os.path.exists(steel_db_path):
        print("\n========== Dataset 1: steel_db ==========")
        print(f"Loading: {steel_db_path}")
        steel_df = pd.read_csv(steel_db_path)
        print(f"  Shape before feature engineering: {steel_df.shape}")

        # Add composition-based features (safe for ML)
        print("\nAdding composition features...")
        steel_df = add_composition_features(steel_df, "steel_db")

        # Add display-only features (NOT for ML)
        print("\nAdding display-only features...")
        steel_df = add_display_features(steel_df, "steel_db")

        # Save the result
        output_path = os.path.join(processed_dir, "steel_db_features.csv")
        steel_df.to_csv(output_path, index=False)
        print(f"\n  Shape after feature engineering: {steel_df.shape}")
        print(f"  Saved to: {output_path}")

        # Print the ML-safe feature list
        ml_features = get_ml_feature_columns("steel_db")
        available_ml = [col for col in ml_features if col in steel_df.columns]
        print(f"\n  ML-safe features ({len(available_ml)}): {available_ml}")
        print(f"  Target columns: {get_target_columns('steel_db')}")
    else:
        print(f"\nWARNING: {steel_db_path} not found — skipping steel_db")

    # ------------------------------------------------------------------
    # Dataset 2: matnavi
    # ------------------------------------------------------------------
    matnavi_path = os.path.join(cleaned_dir, "matnavi_cleaned.csv")
    if os.path.exists(matnavi_path):
        print("\n\n========== Dataset 2: matnavi ==========")
        print(f"Loading: {matnavi_path}")
        matnavi_df = pd.read_csv(matnavi_path)
        print(f"  Shape before feature engineering: {matnavi_df.shape}")

        # Add composition-based features (safe for ML)
        print("\nAdding composition features...")
        matnavi_df = add_composition_features(matnavi_df, "matnavi")

        # Add display-only features (NOT for ML)
        print("\nAdding display-only features...")
        matnavi_df = add_display_features(matnavi_df, "matnavi")

        # Save the result
        output_path = os.path.join(processed_dir, "matnavi_features.csv")
        matnavi_df.to_csv(output_path, index=False)
        print(f"\n  Shape after feature engineering: {matnavi_df.shape}")
        print(f"  Saved to: {output_path}")

        # Print the ML-safe feature list
        ml_features = get_ml_feature_columns("matnavi")
        available_ml = [col for col in ml_features if col in matnavi_df.columns]
        print(f"\n  ML-safe features ({len(available_ml)}): {available_ml}")
        print(f"  Target columns: {get_target_columns('matnavi')}")
    else:
        print(f"\nWARNING: {matnavi_path} not found — skipping matnavi")

    print("\n✅ Feature engineering complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_feature_engineering()
