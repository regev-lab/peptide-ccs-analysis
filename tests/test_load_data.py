from peptide_ccs_analysis import load_data



def test_load_data():
    load_data.DATA1()
    load_data.DATA2()
    SIMULATION_DATA = load_data.SIMULATION_DATA()
    SIMULATION_PEPTIDES = load_data.SIMULATION_PEPTIDES()
    
    
    assert len(SIMULATION_DATA['Gpos_2_SASA_df'].columns) == len(SIMULATION_PEPTIDES['Gpos_2'])
    assert len(SIMULATION_DATA['Gpos_2_helix_content_df'].columns) == len(SIMULATION_PEPTIDES['Gpos_2'])
    assert len(SIMULATION_DATA['Gpos_2_mean_helix_content_vs_residue'].columns) == len(SIMULATION_PEPTIDES['Gpos_2']) + 1
    