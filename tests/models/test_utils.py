from peptide_ccs_analysis.models import utils


def test_get_relative_positions_of_regex_pattern():
    
    assert (
        utils.get_relative_positions_of_pattern('FAVIEWLEFAWEIM', 'A')
        == [ 1 / (14 - 2), 9 / (14 - 2) ]
    )
    
    assert (
        utils.get_relative_positions_of_pattern('KAIMWLKEFAWK', 'K(?!$)')
        == [ 0 / (12 - 2), 6 / (12 - 2) ]
    )
    
    