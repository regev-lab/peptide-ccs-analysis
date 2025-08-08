import itertools

import numpy as np
import pygam

from peptide_ccs_analysis.constants import AA_VOCABULARY
from peptide_ccs_analysis.models import utils
from peptide_ccs_analysis.models.loading import load_model
from peptide_ccs_analysis.models.custom_shared_spline_gam import (
    AbsoluteCCSSharedSplineGAM,
    CCSModeSharedSplineGAM, 
    SplineMultiTerm,
)
    
def test_load_model():
    model = load_model('CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm')
    
    peptides = [
        "IDPQTGWPFFVDHNSR",
        "SLVALGFDLVASGGTATSLRGAGLK",
        "TIQLAGYDVYYEDHDEGTR",
        "ASVCGIFVDFASASHR",
        "SVALYNNPAALLYGDEYVLHK",
        "DALEYVIPELNIHCLR",
        "ASDLDDVEFDLTNHEYAK",
        "AVLTDFESILIEFHK",
        "DVNTEESDLFSTVTLYDNEEIGSLTRQGAK",
        "YFEFTPIELHAITER",
        "LAAPVAAPLAAPVAPVAAPLAAPVAAPLAAPVAAPIATEIVDAHPQYK",
        "AAESTEEVSSLRPPR",
    ]
    
    X = model.transform(peptides)
    assert np.allclose(
        model.predict(X),
        [
            0.2575427138900456,
            0.9747130828515976,
            0.6320115090639251,
            0.9676382988870365,
            0.9658588701462958,
            0.8560584002741651,
            0.6915249387311895,
            0.9892298142908367,
            0.1971354857545545,
            0.934974188348116,
            0.06372442717457799,
            0.96537995183694,
        ]
    )
     
    model = load_model('CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm')
     
    
    peptides = [
        "LIHQGTLQISESIAGNVQK",
        "NTALVLLNQHVSLSFPR",
        "FKDDDGDEEDENGVGDAELR",
        "EINALAEHGDLELDER",
        "ETPAAEATPEPEIPKPEEIPFPK",
        "GYQVATGGTDVHLVLVDVR",
        "AHFDSSEPQLLWDCDNESENSR",
        "EEETAAAPPVEEGEEQKPPAAEELAVDTGK",
        "FAVSQMSALLDSAHLLASSTQR",
        "KPNISGFTDISPEELR",
        "SPVGSGAPQAAAPAPAAHVAGNPGGDAAPAATGTAAAASLATAAGSEDAEK",
        "AAEIIHIGQAIMEQK",
    ]
    
    X = model.transform(peptides)
    assert np.allclose(
        model.predict(X),
        [
            0.5141593823658752,
            0.8447481592628776,
            0.005126450018789478,
            0.23175014474157168,
            0.03761852001607829,
            0.742591314309597,
            0.0005176146922098631,
            0.003300053838513604,
            0.8135242910952611,
            0.14307644247540421,
            1.4991106779187536e-06,
            0.8119141074138269,

        ]
    )
     
    
    model = load_model('AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode')
     
    
    peptides = [
        "FLSCTVEADGIHLVTER",
        "AGGEAGVTLGQPHLSR",
        "FPNVYGIDMPSATELIAHGR",
        "WAYNLSGFNQYGLHR",
        "GLEAALVYVENAHVAGK",
        "ISGWTQALPDMVVSHLFGK",
        "TCNVLLALDQQSPEIAAGVHVNR",
        "SYNILSQDLLEDNSHLYR",
        "LESSQLQIAGLEHLR",
        "FSSELEQIELHNSIR",
        "TITYESPQIDGGAGGDSGTLLTAQTITSESVSTTTTTHITK",
        "AAESTEEVSSLRPPR",
    ]
    
    X = model.transform(peptides)
    assert np.allclose(
        model.predict(X),
        [
            555.6684164793285,
            504.0857681731011,
            590.5925899930131,
            526.5522854096175,
            528.1405431926652,
            572.7913475398364,
            643.0000674835264,
            576.2772102859989,
            526.47142395938,
            526.409416518496,
            838.7531679250391,
            509.27284863671474,
        ]
    )
    
    model = load_model('AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode')
     
    peptides = [
        "VVSEGHTLENCCYQGR",
        "SKGEQIALNVDGACADETSTYSSK",
        "LHGLPEQFLYGTATK",
        "EDIIQGFRYGSDIVPFSK",
        "IDATSASVLASRFDVSGYPTIK",
        "QSQQEAEEEEREEEEEAQIIQR",
        "ARQQDEEMLELPAPAEVAAK",
        "CSNEAVLVATFHPTDPTVLITCGK",
        "NPEDPTEVPGGFLSDLNLASLHVVDAALVDCSVALAK",
        "QEFQFFDEEEETGENHTIFIGPVEK",
        "SPVGSGAPQAAAPAPAAHVAGNPGGDAAPAATGTAAAASLATAAGSEDAEK",
        "AAFYHQPWAQEAVGR",
    ]
    
    X = model.transform(peptides)
    assert np.allclose(
        model.predict(X),
        [
            497.6770847674964,
            572.7575907488522,
            489.91892188732334,
            527.2199335429116,
            563.0593628064573,
            585.2161057472725,
            539.0892628985009,
            594.7689529003297,
            715.5493474128034,
            621.082773281202,
            771.8439186881161,
            484.91005202600803,
        ]
    )
        
    
    
def test_transform():
    
    for model in [
        CCSModeSharedSplineGAM(),
        AbsoluteCCSSharedSplineGAM(),
    ]:
        
        peptides = ['QKDFR', 'NEDFAREFIWLYK']
        X = model.transform(peptides)
        
        assert X.shape == (len(peptides), len(model.features))
        assert X[0, model.features.index('K(?!$)')] == [1/3]
        assert X[1, model.features.index('F')] == [3/11, 7/11]
        
        assert X[0, model.features.index('ind_K$')] == 0
        assert X[1, model.features.index('ind_K$')] == 1
        
        assert X[0, model.features.index('len')] == 5
        assert X[1, model.features.index('len')] == 13
        
        # For each feature that uses the `SharedSpline` term, we check that
        # the feature's corresponding column in X, at a given row (i.e.
        # peptide), contains the list of relative position of that feature in
        # that peptide.
        for i, peptide in enumerate(peptides):
            for feature, term in zip(model.features, model.term_list):
                if isinstance(term, SplineMultiTerm):
                    j = model.features.index(feature)
                    assert X[i,j] == utils.get_relative_positions_of_pattern(peptide, feature)



def test_predict():

    for model in [
        load_model('CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm'),
        load_model('CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm'),
        load_model('AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode'),
        load_model('AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode'),
    ]:    
        for peptide in [        
            "FLSCTVEADGIHLVTER",
            "AGGEAGVTLGQPHLSR",
            "FPNVYGIDMPSATELIAHGR",
            "WAYNLSGFNQYGLHR",
            "GLEAALVYVENAHVAGK",
            "ISGWTQALPDMVVSHLFGK",
            "TCNVLLALDQQSPEIAAGVHVNR",
        ]:
            y_pred = model.predict(model.transform([peptide]))
            
            
            
            M = []
            
            # For each feature, create the entries in `M` that will be multiplied
            # against the model coefficients to produce the GAM's linear prediction.
            for i, (feature, term) in enumerate(zip(model.features, model.gam.terms)):
                if feature in AA_VOCABULARY or feature in ('R(?!$)', 'K(?!$)'):
                    # For features that have shared splines, calculate the b-spline
                    # basis values for each occurence of that feature, and them sum
                    # those values up.
                    
                    relative_positions = utils.get_relative_positions_of_pattern(peptide, feature)
                    if len(relative_positions) == 0:
                        aggregate_b_spline_basis_values = [0] * term.n_splines
                    else:
                        temp = []
                        for relative_position in relative_positions:
                            temp.append(
                                np.array(pygam.utils.b_spline_basis(
                                    relative_position,
                                    edge_knots=term.edge_knots_,
                                    n_splines=term.n_splines,
                                    periodic=False,
                                ).todense())[0]
                            )
                        aggregate_b_spline_basis_values = sum(temp)
                    
                    M.append(aggregate_b_spline_basis_values)
                    
                elif feature == 'len':
                    # For the length feature, calculate the b-spline-basis values from
                    # the length spline.
                    
                    b_spline_basis_values = np.array(pygam.utils.b_spline_basis(
                        len(peptide),
                        edge_knots=term.edge_knots_,
                        n_splines=term.n_splines,
                        periodic=False,
                    ).todense())[0]
                    
                    M.append(b_spline_basis_values)
                    
                elif feature == 'ind_K$':
                    # For the C-terminal indicator feature, append 1 if contains a
                    # C-terminal lysine.
                    M.append([peptide.endswith('K')])
                
            
            M = np.array(list(itertools.chain.from_iterable(M)))
            
            z = sum(M * model.gam.coef_)
            
            if isinstance(model.gam, pygam.LogisticGAM):
                y_pred_other = np.exp(z) / (1 + np.exp(z))
            elif isinstance(model.gam, pygam.LinearGAM):
                y_pred_other = z
            else:
                raise ValueError
            
            assert np.isclose(y_pred, y_pred_other)
            
        
        
        
    
    
    
    
    
    
    
