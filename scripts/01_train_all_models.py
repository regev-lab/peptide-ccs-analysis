from peptide_ccs_analysis import load_data
from peptide_ccs_analysis.models.custom_shared_spline_gam import (
    AbsoluteCCSSharedSplineGAM,
    CCSModeSharedSplineGAM,
)
from peptide_ccs_analysis.models.loading import load_model


def train_ccs_mode_shared_spline_gam_internal_basic_site_near_c_term():
    def process_data(DATA):
        df = DATA.query("charge == 3 & basic_site_count == 3")
        df = df[~df["is_acetylated"]]
        df = df[~df["is_modified"]]
        df = df[df["unmodified_sequence"].apply(lambda x: x[-1] in "RK")]
        df = df[df["len"] >= 15]

        df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) <= 5]

        df["len"] = df["unmodified_sequence"].apply(len)

        sequences = df["unmodified_sequence"]
        y = df["delta_ccs"] > 0

        return sequences, y

    sequences, y = process_data(load_data.DATA1())

    model = CCSModeSharedSplineGAM()

    X = model.transform(sequences)
    model.fit(
        X,
        y,
    )

    saved_model = load_model("CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm")
    assert model == saved_model
    print('Trained the model `CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm` and verified the coefficients match those in `saved_models/`.')
    return model


def train_ccs_mode_shared_spline_gam_internal_basic_site_away_from_c_term():
    def process_data(DATA):
        df = DATA.query("charge == 3 & basic_site_count == 3")
        df = df[~df["is_acetylated"]]
        df = df[~df["is_modified"]]
        df = df[df["unmodified_sequence"].apply(lambda x: x[-1] in "RK")]
        df = df[df["len"] >= 15]

        df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) >= 7]

        df["len"] = df["unmodified_sequence"].apply(len)

        sequences = df["unmodified_sequence"]
        y = df["delta_ccs"] > 0

        return sequences, y

    sequences, y = process_data(load_data.DATA1())

    model = CCSModeSharedSplineGAM()

    X = model.transform(sequences)
    model.fit(
        X,
        y,
    )

    saved_model = load_model("CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm")
    assert model == saved_model
    print('Trained the model `CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm` and verified the coefficients match those in `saved_models/`.')
    return model


def train_absolute_ccs_shared_spline_gam_internal_basic_site_near_c_term_high_mode():
    def process_data(DATA):
        df = DATA.query("charge == 3 & basic_site_count == 3")
        df = df[~df["is_acetylated"]]
        df = df[~df["is_modified"]]
        df = df[df["unmodified_sequence"].apply(lambda x: x[-1] in "RK")]
        df = df[df["len"] >= 15]

        df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) <= 5]

        df["len"] = df["unmodified_sequence"].apply(len)
        df = df[df["delta_ccs"] > 0]

        sequences = df["unmodified_sequence"]
        y = df["ccs"]

        return sequences, y

    sequences, y = process_data(load_data.DATA1())

    model = AbsoluteCCSSharedSplineGAM()

    X = model.transform(sequences)
    model.fit(
        X,
        y,
    )

    saved_model = load_model("AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode")
    assert model == saved_model
    print('Trained the model `AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode` and verified the coefficients match those in `saved_models/`.')
    return model


def train_absolute_ccs_shared_spline_gam_internal_basic_site_away_from_c_term_low_mode():
    def process_data(DATA):
        df = DATA.query("charge == 3 & basic_site_count == 3")
        df = df[~df["is_acetylated"]]
        df = df[~df["is_modified"]]
        df = df[df["unmodified_sequence"].apply(lambda x: x[-1] in "RK")]
        df = df[df["len"] >= 15]

        df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) >= 7]

        df["len"] = df["unmodified_sequence"].apply(len)
        df = df[df["delta_ccs"] <= 0]

        sequences = df["unmodified_sequence"]
        y = df["ccs"]

        return sequences, y

    sequences, y = process_data(load_data.DATA1())

    model = AbsoluteCCSSharedSplineGAM()

    X = model.transform(sequences)
    model.fit(
        X,
        y,
    )

    saved_model = load_model("AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode")
    assert model == saved_model
    print('Trained the model `AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode` and verified the coefficients match those in `saved_models/`.')
    return model


train_ccs_mode_shared_spline_gam_internal_basic_site_near_c_term()
train_ccs_mode_shared_spline_gam_internal_basic_site_away_from_c_term()
train_absolute_ccs_shared_spline_gam_internal_basic_site_near_c_term_high_mode()
train_absolute_ccs_shared_spline_gam_internal_basic_site_away_from_c_term_low_mode()
