from peptide_ccs_analysis import constants


def test_constants():
    
    assert constants.DATA_PATH.exists()
    assert constants.DATA_PATH.resolve() == constants.DATA_PATH
    assert constants.SAVED_MODELS_PATH.exists()
    assert constants.SAVED_MODELS_PATH.resolve() == constants.SAVED_MODELS_PATH
    
    assert len(constants.AA_VOCABULARY) == 20