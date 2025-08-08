import ast  # TODO: Remove ast import
import functools

import pandas as pd

from peptide_ccs_analysis.constants import DATA_PATH

kde_svm_slope = 0.12979868744140854
kde_svm_intercept = 278.90332472949825


def kde_svm_separator(mass):
    return kde_svm_slope * mass + kde_svm_intercept


def kde_svm_dist(df):
    return df["ccs"] - (kde_svm_slope * df["mass"] + kde_svm_intercept)


@functools.cache
def DATA1():
    DATA1 = pd.read_parquet(DATA_PATH / "external/MeierTrainData.parquet")
    DATA1["basic_sites"] = DATA1["basic_sites"].apply(ast.literal_eval)
    DATA1["basic_site_positions"] = DATA1["basic_site_positions"].apply(ast.literal_eval)

    DATA1["delta_ccs"] = kde_svm_dist(DATA1)

    return DATA1


@functools.cache
def DATA2():
    DATA2 = pd.read_parquet(DATA_PATH / "external/MeierTestData.parquet")
    DATA2["basic_sites"] = DATA2["basic_sites"].apply(ast.literal_eval)
    DATA2["basic_site_positions"] = DATA2["basic_site_positions"].apply(ast.literal_eval)

    DATA2["delta_ccs"] = kde_svm_dist(DATA2)

    return DATA1


@functools.cache
def SIMULATION_DATA():
    SIMULATION_DATA = {}

    SIMULATION_DATA["Gpos_1_SASA_df"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_1_SASA_df.csv", index_col=0
    )
    SIMULATION_DATA["Gpos_2_SASA_df"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_2_SASA_df.csv", index_col=0
    )
    SIMULATION_DATA["Gpos_1_helix_content_df"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_1_helix_content_df.csv", index_col=0
    )
    SIMULATION_DATA["Gpos_2_helix_content_df"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_2_helix_content_df.csv", index_col=0
    )
    SIMULATION_DATA["Gpos_1_mean_helix_content_vs_residue"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_1_mean_helix_content_vs_residue.csv", index_col=0
    )
    SIMULATION_DATA["Gpos_2_mean_helix_content_vs_residue"] = pd.read_csv(
        DATA_PATH / "raw/Gpos_2_mean_helix_content_vs_residue.csv", index_col=0
    )

    # Change units from nm^2 to Angstrom^2
    SIMULATION_DATA["Gpos_1_SASA_df"] *= 100
    SIMULATION_DATA["Gpos_2_SASA_df"] *= 100

    index_to_remove = 5

    SIMULATION_DATA["Gpos_2_SASA_df"].drop(columns=f"position_{index_to_remove}", inplace=True)
    SIMULATION_DATA["Gpos_2_helix_content_df"].drop(
        columns=f"position_{index_to_remove}", inplace=True
    )
    SIMULATION_DATA["Gpos_2_mean_helix_content_vs_residue"].drop(
        columns=f"position_{index_to_remove}", inplace=True
    )

    return SIMULATION_DATA


@functools.cache
def SIMULATION_PEPTIDES():
    SIMULATION_PEPTIDES = {}

    base_sequence = "NQVSLLNVVMDLKK"
    peptides = []
    for i in range(len(base_sequence)):
        peptides.append(base_sequence[:i] + "GGG" + base_sequence[i:])
    SIMULATION_PEPTIDES["Gpos_1"] = peptides

    base_sequence = "LAHVGFDNATFLSER"
    peptides = []
    for i in range(len(base_sequence)):
        peptides.append(base_sequence[:i] + "GGG" + base_sequence[i:])
    SIMULATION_PEPTIDES["Gpos_2"] = peptides

    assert SIMULATION_PEPTIDES["Gpos_2"][4] == SIMULATION_PEPTIDES["Gpos_2"][5]

    index_to_remove = 5
    del SIMULATION_PEPTIDES["Gpos_2"][index_to_remove]

    assert len(SIMULATION_PEPTIDES["Gpos_1"]) == len(set(SIMULATION_PEPTIDES["Gpos_1"]))
    assert len(SIMULATION_PEPTIDES["Gpos_2"]) == len(set(SIMULATION_PEPTIDES["Gpos_2"]))

    return SIMULATION_PEPTIDES
