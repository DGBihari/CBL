"""
Builds the E_data matrix for coefficient fitting -> import to coefficients_fitting.py

Produces the following matrices and lists:
--------
E_data    : np.ndarray (K_months, NR)  — monthly crime counts per PFA
months    : list[str]  sorted e.g. ['2021-01', '2021-02', .., '2025-12']
pfa_names : list[str]  PFA names col
---------
"""

import os
import glob
import numpy as np
import pandas as pd
LAD_TO_PFA_LOOKUP = os.path.join(os.path.dirname(__file__), "csv", "lad_to_pfa_lookup.csv")

# make it consistent through the codebase, put it in a header or whatever ltr
def get_master_pfa_list_from_lookup(lookup_csv=LAD_TO_PFA_LOOKUP):
    
    #Reads the master PFA list from lad_to_pfa_lookup.csv (PFA24NM column, unique, order of appearance).
    df = pd.read_csv(lookup_csv, dtype=str)
    # Standardize PFA names for consistency
    df["PFA24NM"] = standardize_pfa_names(df["PFA24NM"])
    pfa_names = df["PFA24NM"].drop_duplicates().tolist()
    return pfa_names


CRIME_CSV_DIR = r"C:\Users\danie\Downloads\uni\year 2\q4\4CBLW020\project\CBL\csv\crime_data"   # change

# Aggressive Sanitization Function, almost same as run_sde_pipeline to make it global, use mask for pandas?
def standardize_pfa_names(series):
    series = series.astype(str).str.strip().str.replace('\n', '', regex=True)
    # Strip out the formal suffixes from the raw UK Police data
    series = series.str.replace(' Police', '', regex=False)
    series = series.str.replace(' Constabulary', '', regex=False)
    series = series.str.replace(' Service', '', regex=False)
    # The Two Londons Fix:
    series.loc[series.str.contains('Metropolitan', case=False, na=False)] = 'Metropolitan Police'
    # Force the crime data's "City of London" to match the map's "London, City of"
    series.loc[series.str.contains('City of London', case=False, na=False)] = 'London, City of'
    # fioxes
    series.loc[series.str.contains('Devon', case=False, na=False)] = 'Devon & Cornwall' # do correct mapping for these, then series.replace?
    series.loc[series.str.contains('Hampshire', case=False, na=False)] = 'Hampshire'
    #series.loc[series.str.contains('Greater Manchester', case=False, na=False)] = 'Greater Manchester'
    return series


def build_E_data(directory: str = CRIME_CSV_DIR):
    """
    Recursively finds all *-street.csv files under `directory`,
    counts crimes per (Month, PFA), and returns a (K_months, NR) matrix.
    """
    # ** means any subdirectory depth — handles YYYY-MM/YYYY-MM-force-street.csv
    files = sorted(glob.glob(os.path.join(directory, "**", "*-street.csv"), recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No *-street.csv files found under {directory}.\n"
            "Check that CRIME_CSV_DIR points to your crime data folder."
        )

    print(f"Found {len(files)} CSV files across all month folders.")

    # Read only the 3 columns we need, 43 forces × 60 months
    # to do: remove :LSOA code if not needed, improved performance
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(
                             f, 
                             usecols=["Month", "Falls within", "LSOA code"],
                             dtype=str)
            chunks.append(df)
        except Exception as e:
            print(f"  ! Skipped {os.path.basename(f)}: {e}")

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.dropna(subset=["Falls within", "Month"])

    print(f"Total crime records loaded: {len(raw):,}")

    # Normalise PFA names using standardize_pfa_names
    raw["PFA"] = standardize_pfa_names(raw["Falls within"])

    # Count incidents per (month, PFA)
    counts = (
        raw.groupby(["Month", "PFA"])
           .size()
           .reset_index(name="crime_count")
    )

    # Pivot → matrix
    pivot = (
        counts.pivot(index="Month", columns="PFA", values="crime_count")
              .fillna(0)
              .sort_index()
    )

    months    = pivot.index.tolist()       # ['2021-01', '2021-02', ..., '2025-12']
    pfa_names = pivot.columns.tolist()
    E_data    = pivot.values.astype(float) # (K_months, NR)

    print(f"\nE_data matrix shape : {E_data.shape}  ({len(months)} months * {len(pfa_names)} PFAs)")
    print(f"Months: {months[0]}  →  {months[-1]}")
    print(f"Mean monthly crimes per PFA: {E_data.mean():.1f}")


    return E_data, months, pfa_names

#Reorder E_data columns to match needed matrix by Salih's model aka master_pfa_names
def align_to_master_pfa_list(E_data, e_pfa_names, master_pfa_names):
    # A[row_selection, column_selection], ":" means everything
    K  = E_data.shape[0] # months
    NR = len(master_pfa_names) # PFAS
    E_aligned = np.zeros((K, NR)) # E_aligned = np.zeros((num_months, num_pfas))
    e_idx = {name: i for i, name in enumerate(e_pfa_names)} # pfa_to_column_index, lookup dict with indexes so "A&S":0 etc
    # eg colunm 0 Avon&Somerset
    for j, name in enumerate(master_pfa_names):
        if name in e_idx:
            E_aligned[:, j] = E_data[:, e_idx[name]] # copy column from E_data to E_aligned based on index
        else:
            print(f"  !!!'{name}' not in crime data — column set to 0")

    return E_aligned


if __name__ == "__main__":
    print("Starting" )
    E_data, months, pfa_names = build_E_data()

    print("\nPer-PFA mean monthly crime counts:")
    # Get master PFA list from lookup CSV
    master_pfa_names = get_master_pfa_list_from_lookup()

    # Align E_data to master PFA list
    E_data_aligned = align_to_master_pfa_list(E_data, pfa_names, master_pfa_names)

    print("\nPer-PFA mean monthly crime counts (aligned):")

    # Mean crime count for each PFA across all months
    mean_crime_counts = E_data_aligned.mean(axis=0)

    # Pair each PFA name with its mean count
    pfa_mean_pairs = list(zip(master_pfa_names, mean_crime_counts))

    # Sort descending by mean crime count
    pfa_mean_pairs.sort(key=lambda pair: pair[1], reverse=True)

    # Print results
    for pfa_name, mean_count in pfa_mean_pairs:
        print(f"  {pfa_name:<35} {mean_count:>8.0f}")