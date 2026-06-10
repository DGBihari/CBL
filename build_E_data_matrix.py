"""
Builds the E_data matrix for coefficient fitting -> import to coefficients_fitting.py

"""

import os
import glob
import numpy as np
import pandas as pd
LAD_TO_PFA_LOOKUP = os.path.join(os.path.dirname(__file__), "csv", "lad_to_pfa_lookup.csv")

# make it consistent through the codebase, put it in a header or whatever ltr
def get_master_pfa_list_from_lookup(lookup_csv=LAD_TO_PFA_LOOKUP):
    
    df = pd.read_csv(lookup_csv, dtype=str)
    df["PFA24NM"] = standardize_pfa_names(df["PFA24NM"])
    pfa_names = df["PFA24NM"].drop_duplicates().tolist()
    return pfa_names


CRIME_CSV_DIR = os.path.join(os.path.dirname(__file__), "csv", "crime_data")

def standardize_pfa_names(series):
    series = series.astype(str).str.strip().str.replace('\n', '', regex=True)
    series = series.str.replace(' Police', '', regex=False)
    series = series.str.replace(' Constabulary', '', regex=False)
    series = series.str.replace(' Service', '', regex=False)
    series.loc[series.str.contains('Metropolitan', case=False, na=False)] = 'Metropolitan Police'
    series.loc[series.str.contains('City of London', case=False, na=False)] = 'London, City of'
    # fioxes
    series.loc[series.str.contains('Devon', case=False, na=False)] = 'Devon and Cornwall' # do correct mapping for these, then series.replace?
    series.loc[series.str.contains('Hampshire', case=False, na=False)] = 'Hampshire and Isle of Wight'
    return series

TARGET_CRIMES = ['Anti-social behaviour', 'Violence and sexual offences']

def build_E_data(directory: str = CRIME_CSV_DIR):
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
                             usecols=["Month", "Falls within", "LSOA code", "Crime type"],
                             dtype=str)
            df = df[df["Crime type"].isin(TARGET_CRIMES)]
            chunks.append(df)
        except Exception as e:
            print(f"  ! Skipped {os.path.basename(f)}: {e}")

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.dropna(subset=["Falls within", "Month"])


    raw["PFA"] = standardize_pfa_names(raw["Falls within"])

    counts = (
        raw.groupby(["Month", "PFA"])
           .size()
           .reset_index(name="crime_count")
    )

    pivot = (
        counts.pivot(index="Month", columns="PFA", values="crime_count")
              .fillna(0)
              .sort_index()
    )

    months    = pivot.index.tolist()       # ['2021-01', '2021-02', ..., '2025-12']
    pfa_names = pivot.columns.tolist()
    E_data    = pivot.values.astype(float) # (K_months, NR)



    return E_data, months, pfa_names

def align_to_master_pfa_list(E_data, e_pfa_names, master_pfa_names):
    K  = E_data.shape[0] # months
    NR = len(master_pfa_names) 
    E_aligned = np.zeros((K, NR)) 
    e_idx = {name: i for i, name in enumerate(e_pfa_names)} 
    # eg colunm 0 Avon&Somerset
    for j, name in enumerate(master_pfa_names):
        if name in e_idx:
            E_aligned[:, j] = E_data[:, e_idx[name]] # copy column from E_data to E_aligned based on index
        else:
            print(f"  !!!'{name}' not in crime data — column set to 0")

    return E_aligned

# For testing, always run pipeline to actually run the main functionalities
if __name__ == "__main__":
    E_data, months, pfa_names = build_E_data()

    master_pfa_names = get_master_pfa_list_from_lookup()

    E_data_aligned = align_to_master_pfa_list(E_data, pfa_names, master_pfa_names)


    mean_crime_counts = E_data_aligned.mean(axis=0)

    pfa_mean_pairs = list(zip(master_pfa_names, mean_crime_counts))

    pfa_mean_pairs.sort(key=lambda pair: pair[1], reverse=True)

    for pfa_name, mean_count in pfa_mean_pairs:
        print(f"  {pfa_name:<35} {mean_count:>8.0f}")