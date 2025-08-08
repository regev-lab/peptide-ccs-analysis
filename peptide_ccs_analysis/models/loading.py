import functools

from peptide_ccs_analysis.constants import SAVED_MODELS_PATH
from peptide_ccs_analysis.models.custom_shared_spline_gam import (
    AbsoluteCCSSharedSplineGAM,
    CCSModeSharedSplineGAM,
)


@functools.cache
def load_model(model_name):
    match model_name:
        case "CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm":
            model = CCSModeSharedSplineGAM().load_params(
                SAVED_MODELS_PATH / "CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm_params.npz"
            )
            return model

        case "CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm":
            model = CCSModeSharedSplineGAM().load_params(
                SAVED_MODELS_PATH
                / "CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm_params.npz"
            )
            return model

        case "AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode":
            model = AbsoluteCCSSharedSplineGAM().load_params(
                SAVED_MODELS_PATH
                / "AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode_params.npz"
            )
            return model

        case "AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode":
            model = AbsoluteCCSSharedSplineGAM().load_params(
                SAVED_MODELS_PATH
                / "AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode_params.npz"
            )
            return model
