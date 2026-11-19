"""
data_preprocessing.py — MaterialMind AI
Cleans and prepares both raw datasets for analysis and modeling.

Datasets:
  1. Steel Database (Excel) — 3,234 steel alloy records with composition and mechanical properties
  2. MatNavi Mechanical Properties (CSV) — 915 records of steel mechanical properties

Author: MaterialMind AI Team
"""

import os
import pandas as pd


# ──────────────────────────────────────────────
#  STEEL DATABASE FUNCTIONS
# ──────────────────────────────────────────────

def load_steel_db(filepath):
    """Load the Excel steel database and return raw DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw Excel file.

    Returns
    -------
    pd.DataFrame
        Raw data exactly as read from the Excel file.
    """
    print(f"Loading steel database from: {filepath}")
    df = pd.read_excel(filepath)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_steel_db(df):
    """Clean the steel database: drop empty/duplicate cols, standardize names.

    Steps:
      1. Drop 4 empty spacer columns (Unnamed: 3, 5, 24, 28)
      2. Drop the duplicate cluster column 'Cluster Number (0 to 11)'
      3. Drop the 'Entry' column (just a row index, not useful)
      4. Standardize all column names to lowercase with underscores
      5. Verify no missing values or duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw steel database DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for analysis.
    """
    print("\nCleaning steel database...")
    df = df.copy()  # Don't modify the original

    # Step 1: Drop empty spacer columns
    empty_cols = ["Unnamed: 3", "Unnamed: 5", "Unnamed: 24", "Unnamed: 28"]
    df = df.drop(columns=empty_cols)
    print(f"  Dropped {len(empty_cols)} empty spacer columns")

    # Step 2: Drop the duplicate cluster column
    df = df.drop(columns=["Cluster Number (0 to 11)"])
    print("  Dropped duplicate cluster column 'Cluster Number (0 to 11)'")

    # Step 3: Drop the Entry column (just a row index)
    df = df.drop(columns=["Entry"])
    print("  Dropped 'Entry' column (row index)")

    # Step 4: Standardize column names to lowercase with underscores
    # Build a mapping from old names to new clean names
    column_name_map = {
        "Name": "name",
        "Processing condition": "processing_condition",
        "Al": "al",
        "Cu": "cu",
        "Mn": "mn",
        "N": "n",
        "Ni": "ni",
        "Ti": "ti",
        "S": "s",
        "Fe": "fe",
        "Zr": "zr",
        "P": "p",
        "Si": "si",
        "V": "v",
        "Mo": "mo",
        "Co": "co",
        "C": "c",
        "Nb": "nb",
        "B": "b",
        "Cr": "cr",
        "(Ultimate) Tensile strength (MPa)": "tensile_strength_mpa",
        "Yield strength (MPa)": "yield_strength_mpa",
        "Ductility (%)": "ductility_pct",
        "Cluster Number": "cluster_number",
        "Cluster label": "cluster_label",
    }
    df = df.rename(columns=column_name_map)
    print(f"  Standardized {len(column_name_map)} column names")

    # Step 5: Verify data quality
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"  Missing values: {missing_count}")
    print(f"  Duplicate rows: {duplicate_count}")

    print(f"  Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
#  MATNAVI DATASET FUNCTIONS
# ──────────────────────────────────────────────

def load_matnavi(filepath):
    """Load the MatNavi CSV file and return raw DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data exactly as read from the CSV file.
    """
    print(f"\nLoading MatNavi dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_matnavi(df):
    """Clean MatNavi data: strip column names, standardize, flag outliers.

    Steps:
      1. Strip leading/trailing whitespace from all column names
      2. Standardize column names to lowercase with underscores
      3. Flag the known outlier at row 626 (tensile_strength_mpa = 6661)
      4. Verify no missing values or duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw MatNavi DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an 'is_outlier' column added.
    """
    print("\nCleaning MatNavi dataset...")
    df = df.copy()  # Don't modify the original

    # Step 1: Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("  Stripped leading/trailing whitespace from column names")

    # Step 2: Standardize column names to lowercase with underscores
    column_name_map = {
        "Alloy code": "alloy_code",
        "C": "c",
        "Si": "si",
        "Mn": "mn",
        "P": "p",
        "S": "s",
        "Ni": "ni",
        "Cr": "cr",
        "Mo": "mo",
        "Cu": "cu",
        "V": "v",
        "Al": "al",
        "N": "n",
        "Ceq": "ceq",
        "Nb + Ta": "nb_ta",
        "Temperature (°C)": "temperature_c",
        "0.2% Proof Stress (MPa)": "proof_stress_mpa",
        "Tensile Strength (MPa)": "tensile_strength_mpa",
        "Elongation (%)": "elongation_pct",
        "Reduction in Area (%)": "reduction_in_area_pct",
    }
    df = df.rename(columns=column_name_map)
    print(f"  Standardized {len(column_name_map)} column names")

    # Step 3: Flag the known outlier
    # Row 626 has tensile_strength_mpa = 6661 — almost certainly a data entry error
    # (likely should be 666.1 MPa). We flag it but keep the row.
    df["is_outlier"] = False
    outlier_mask = df["tensile_strength_mpa"] == 6661
    outlier_count = outlier_mask.sum()
    df.loc[outlier_mask, "is_outlier"] = True

    if outlier_count > 0:
        print(f"  ⚠ WARNING: Found {outlier_count} outlier row(s) with "
              f"tensile_strength_mpa = 6661 (likely data entry error)")
        print(f"    Row index: {df.index[outlier_mask].tolist()}")
        print(f"    Flagged with is_outlier = True (row kept in dataset)")

    # Step 4: Verify data quality
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"  Missing values: {missing_count}")
    print(f"  Duplicate rows: {duplicate_count}")

    print(f"  Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def save_cleaned_data(df, filepath):
    """Save a cleaned DataFrame to CSV.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to save.
    filepath : str
        Output path for the CSV file.
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"  Saved to: {filepath} ({df.shape[0]} rows, {df.shape[1]} columns)")


# ──────────────────────────────────────────────
#  MAIN PREPROCESSING PIPELINE
# ──────────────────────────────────────────────

def run_preprocessing():
    """Main function: load, clean, and save both datasets.

    Prints summary statistics for each cleaned dataset.
    """
    # Set up file paths (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    steel_raw_path = os.path.join(project_root, "data", "raw",
                                  "steel_database_with_labelled_clusters.xlsx")
    matnavi_raw_path = os.path.join(project_root, "data", "raw",
                                    "MatNavi_mechanical_properties.csv")
    steel_clean_path = os.path.join(project_root, "data", "processed",
                                    "steel_db_cleaned.csv")
    matnavi_clean_path = os.path.join(project_root, "data", "processed",
                                      "matnavi_cleaned.csv")

    print("=" * 60)
    print("  MaterialMind AI — Data Preprocessing Pipeline")
    print("=" * 60)

    # ── Dataset 1: Steel Database ──
    steel_raw = load_steel_db(steel_raw_path)
    steel_clean = clean_steel_db(steel_raw)
    save_cleaned_data(steel_clean, steel_clean_path)

    # Print summary statistics for the steel dataset
    print("\n--- Steel Database Summary ---")
    print(f"  Columns: {list(steel_clean.columns)}")
    # Show stats for the numeric mechanical property columns
    mech_cols = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
    print(steel_clean[mech_cols].describe().round(2).to_string())

    # ── Dataset 2: MatNavi ──
    matnavi_raw = load_matnavi(matnavi_raw_path)
    matnavi_clean = clean_matnavi(matnavi_raw)
    save_cleaned_data(matnavi_clean, matnavi_clean_path)

    # Print summary statistics for the MatNavi dataset
    print("\n--- MatNavi Dataset Summary ---")
    print(f"  Columns: {list(matnavi_clean.columns)}")
    mech_cols_matnavi = ["proof_stress_mpa", "tensile_strength_mpa",
                         "elongation_pct", "reduction_in_area_pct"]
    print(matnavi_clean[mech_cols_matnavi].describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)


# ──────────────────────────────────────────────
#  RUN AS SCRIPT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()
