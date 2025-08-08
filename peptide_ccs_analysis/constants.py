from pathlib import Path

import matplotlib as mpl
import numpy as np

### Paths

DATA_PATH = Path(__file__).parents[1] / "datasets"
SAVED_MODELS_PATH = Path(__file__).parents[1] / "saved_models"

### Modle constants

AA_VOCABULARY = "QNAVILFMYWPGCSTRKHED"

### Figure constants

PPI = 72

HEXBIN_XLIM = [800, 4780]
HEXBIN_YLIM = [380, 920]

GRAY = "#404040"

DENSITY_CMAP = mpl.colormaps["YlGn_r"]
SCATTER_COLOR = DENSITY_CMAP(0.3)

DELTA_CCS_CMAP = "RdYlBu_r"

DISCRETIZED_SEPARATION_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "", mpl.colormaps[DELTA_CCS_CMAP](np.linspace(0.1, 0.9, 100))
)

GAM_SPLINE_LOWER_MODE_COLOR = DISCRETIZED_SEPARATION_CMAP(0.0)
GAM_SPLINE_UPPER_MODE_COLOR = DISCRETIZED_SEPARATION_CMAP(1.0)
