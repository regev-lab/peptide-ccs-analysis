import functools

import numpy as np
import pandas as pd
import pygam
from pygam.utils import b_spline_basis, gen_edge_knots
import scipy

from peptide_ccs_analysis.constants import AA_VOCABULARY
from peptide_ccs_analysis.models import utils


class AbsoluteCCSSharedSplineGAM:
    def __init__(self):
        self.features = (
            [aa for aa in AA_VOCABULARY if aa not in "RKH"]
            + ["R(?!$)", "K(?!$)", "H"]
            + ["len"]
            + ["ind_K$"]
        )

        n_splines = 10
        spline_kwargs = {}
        spline_kwargs["n_splines"] = n_splines
        spline_kwargs["penalties"] = ("derivative", "l2")
        spline_kwargs["lam"] = (0.6 * (n_splines / 20) ** 2, 1e-4 * n_splines / 20)

        self.term_list = []
        for i, feature in enumerate(self.features):
            if feature in AA_VOCABULARY or feature in ["R(?!$)", "K(?!$)"]:
                term = SplineMultiTerm(i, **spline_kwargs)
            elif feature == "ind_K$":
                term = pygam.terms.LinearTerm(i, lam=0.0)
            elif feature == "len":
                term = pygam.terms.SplineTerm(
                    i,
                    penalties="derivative",
                )
            else:
                raise ValueError

            self.term_list.append(term)

        terms = functools.reduce(lambda a, b: a + b, self.term_list)

        self.gam = pygam.LinearGAM(terms, fit_intercept=False)

        # Initialize parameters usually initialized by `_validate_params` and
        # `fit` so we can call `predict` without first calling `fit`. Namely,
        # this needed since we directly load the coefficients from a csv file
        # into a freshly instantiated model, in `load_model`.
        self.gam.link = pygam.links.LINKS[self.gam.link]()

        self.gam.statistics_ = {}
        self.gam.statistics_["m_features"] = len(self.features)

    def transform(self, peptides):
        if not len(peptides):
            return []

        # Create the transformed data, X, column by column (i.e., feature by feature).
        X = []
        for feature, term in zip(self.features, self.term_list):
            if isinstance(term, SplineMultiTerm):
                feature_column = [
                    utils.get_relative_positions_of_pattern(peptide, feature)
                    for peptide in peptides
                ]
            elif feature == "ind_K$":
                feature_column = [int(peptide.endswith("K")) for peptide in peptides]

            elif feature == "len":
                feature_column = [len(peptide) for peptide in peptides]

            X.append(pd.Series(feature_column))
            # Casting to `pd.Series` first is a trick to force the `np.stack`
            # below to produce a (ragged) array of lists.
            # Without `pd.Series`, numpy will sometimes try to convert the
            # nested lists into another dimension.

        X = np.stack(X, axis=1)
        # This is a 2D numpy array containg a mix of lists and integrers.

        return X

    def predict(self, X):
        with disable_pygam_checks():
            y_pred = self.gam.predict(X)
        return y_pred

    def fit(self, X, y, **kwargs):
        with disable_pygam_checks():
            self.gam.fit(X, y / 25, **kwargs)

        self.gam.coef_ = 25 * self.gam.coef_
        self.gam.statistics_["cov"] = 25**2 * self.gam.statistics_["cov"]
        return self

    def load_params(self, file_path):
        params = np.load(file_path)

        coef = params["coef"]
        terms_edge_knots = params["terms_edge_knots"]
        statistics_cov = params["statistics_cov"]
        statistics_n_samples = params["statistics_n_samples"]
        statistics_edof = params["statistics_edof"]

        self.gam.coef_ = coef

        for i, term in enumerate(self.gam.terms):
            term.edge_knots_ = terms_edge_knots[i, :]

        self.gam.statistics_["cov"] = statistics_cov
        self.gam.statistics_["n_samples"] = statistics_n_samples
        self.gam.statistics_["edof"] = statistics_edof

        return self

    def save_params(self, file_path):
        coef = self.gam.coef_
        terms_edge_knots = self.gam.terms.edge_knots_
        statistics_cov = self.gam.statistics_["cov"]
        statistics_n_samples = self.gam.statistics_["n_samples"]
        statistics_edof = self.gam.statistics_["edof"]

        np.savez(
            file_path,
            coef=coef,
            terms_edge_knots=terms_edge_knots,
            statistics_cov=statistics_cov,
            statistics_n_samples=statistics_n_samples,
            statistics_edof=statistics_edof,
        )

    def __eq__(self, other):
        equal = True
        equal &= np.allclose(self.gam.coef_, other.gam.coef_)
        equal &= np.allclose(self.gam.terms.edge_knots_, other.gam.terms.edge_knots_)
        equal &= np.allclose(self.gam.statistics_["cov"], other.gam.statistics_["cov"])
        equal &= np.allclose(self.gam.statistics_["edof"], other.gam.statistics_["edof"])
        return equal


class CCSModeSharedSplineGAM:
    def __init__(self):
        self.features = (
            [aa for aa in AA_VOCABULARY if aa not in "RKH"]
            + ["R(?!$)", "K(?!$)", "H"]
            + ["len"]
            + ["ind_K$"]
        )

        n_splines = 10
        spline_kwargs = {}
        spline_kwargs["n_splines"] = n_splines
        spline_kwargs["penalties"] = ("derivative", "l2")
        spline_kwargs["lam"] = (0.6 * (n_splines / 20) ** 2, 1e-4 * n_splines / 20)

        self.term_list = []
        for i, feature in enumerate(self.features):
            if feature in AA_VOCABULARY or feature in ["R(?!$)", "K(?!$)"]:
                term = SplineMultiTerm(i, **spline_kwargs)
            elif feature == "ind_K$":
                term = pygam.terms.LinearTerm(i, lam=0.0)
            elif feature == "len":
                term = pygam.terms.SplineTerm(i, penalties="derivative")
            else:
                raise ValueError

            self.term_list.append(term)

        terms = functools.reduce(lambda a, b: a + b, self.term_list)

        self.gam = pygam.LogisticGAM(terms, fit_intercept=False)

        # Initialize parameters usually initialized by `_validate_params` and
        # `fit` so we can call `predict` without first calling `fit`. Namely,
        # this needed since we directly load the coefficients from a csv file
        # into a freshly instantiated model, in `load_model`.
        self.gam.link = pygam.links.LINKS[self.gam.link]()
        self.gam.distribution = pygam.distributions.DISTRIBUTIONS[self.gam.distribution]()

        self.gam.statistics_ = {}
        self.gam.statistics_["m_features"] = len(self.features)

    def transform(self, peptides):
        if not len(peptides):
            return []

        # Create the transformed data, X, column by column (i.e., feature by feature).
        X = []
        for feature, term in zip(self.features, self.term_list):
            if isinstance(term, SplineMultiTerm):
                feature_column = [
                    utils.get_relative_positions_of_pattern(peptide, feature)
                    for peptide in peptides
                ]
            elif feature == "ind_K$":
                feature_column = [int(peptide.endswith("K")) for peptide in peptides]

            elif feature == "len":
                feature_column = [len(peptide) for peptide in peptides]

            X.append(pd.Series(feature_column))
            # Casting to `pd.Series` first is a trick to force the `np.stack`
            # below to produce a (ragged) array of lists.
            # Without `pd.Series`, numpy will sometimes try to convert the
            # nested lists into another dimension.

        X = np.stack(X, axis=1)
        # This is a 2D numpy array containg a mix of lists and integrers.

        return X

    def predict(self, X):
        with disable_pygam_checks():
            y_pred = self.gam.predict_mu(X)
        return y_pred

    def fit(self, X, y):
        with disable_pygam_checks():
            self.gam.fit(X, y)
        return self

    def load_params(self, file_path):
        params = np.load(file_path)

        coef = params["coef"]
        terms_edge_knots = params["terms_edge_knots"]
        statistics_cov = params["statistics_cov"]
        statistics_n_samples = params["statistics_n_samples"]
        statistics_edof = params["statistics_edof"]

        self.gam.coef_ = coef

        for i, term in enumerate(self.gam.terms):
            term.edge_knots_ = terms_edge_knots[i, :]

        self.gam.statistics_["cov"] = statistics_cov
        self.gam.statistics_["n_samples"] = statistics_n_samples
        self.gam.statistics_["edof"] = statistics_edof

        return self

    def save_params(self, file_path):
        coef = self.gam.coef_
        terms_edge_knots = self.gam.terms.edge_knots_
        statistics_cov = self.gam.statistics_["cov"]
        statistics_n_samples = self.gam.statistics_["n_samples"]
        statistics_edof = self.gam.statistics_["edof"]

        np.savez(
            file_path,
            coef=coef,
            terms_edge_knots=terms_edge_knots,
            statistics_cov=statistics_cov,
            statistics_n_samples=statistics_n_samples,
            statistics_edof=statistics_edof,
        )

    def __eq__(self, other):
        equal = True
        equal &= np.allclose(self.gam.coef_, other.gam.coef_)
        equal &= np.allclose(self.gam.terms.edge_knots_, other.gam.terms.edge_knots_)
        equal &= np.allclose(self.gam.statistics_["cov"], other.gam.statistics_["cov"])
        equal &= np.allclose(self.gam.statistics_["edof"], other.gam.statistics_["edof"])
        return equal


class SplineMultiTerm(pygam.pygam.SplineTerm):
    def __init__(
        self,
        feature,
        n_splines=20,
        spline_order=3,
        lam=0.6,
        penalties="auto",
        constraints=None,
        dtype="numerical",
        basis="ps",
        by=None,
        edge_knots=None,
        verbose=False,
    ):
        self.basis = basis
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.by = by
        self._name = "spline_term"
        self._minimal_name = "s_multi"

        if edge_knots is not None:
            self.edge_knots_ = edge_knots

        super(pygam.pygam.SplineTerm, self).__init__(
            feature=feature,
            lam=lam,
            penalties=penalties,
            constraints=constraints,
            fit_linear=False,
            fit_splines=True,
            dtype=dtype,
            verbose=verbose,
        )

        self._exclude += ["fit_linear", "fit_splines"]

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        if self.feature >= X.shape[1]:
            raise ValueError(
                "term requires feature {}, but X has only {} dimensions".format(
                    self.feature, X.shape[1]
                )
            )

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError(
                "by variable requires feature {}, but X has only {} dimensions".format(
                    self.by, X.shape[1]
                )
            )

        if not hasattr(self, "edge_knots_"):
            self.edge_knots_ = gen_edge_knots(
                np.concatenate(X[:, self.feature]), self.dtype, verbose=verbose
            )
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows and `n_splines` columns
        """

        if X[:, self.feature].dtype == "O":
            # If the `X[:, self.feature]` column is dtype 'O', then it contains
            # a list for each row. These lists are the list of feature values
            # for each row, and can be of different length.
            # We ultimately want to calculate, for each row, the sum of
            # b-spline basis values for all feature values in the list.

            # To do so, we first flatten the column by concatenating all
            # the lists together into a "super" list, and calculate the
            # b-spline basis values for each element in the super list.
            # Lastly, we unflatten and sum the b-spline basis values based on
            # which list they originated from. This is done by keeping the
            # lengths of each original list in the variable `counts`.

            X_feature = X[:, self.feature]
            counts = map(len, X_feature)
            flattened_X_feature = np.concatenate(X_feature)

            flattened_spline_values = b_spline_basis(
                flattened_X_feature,
                edge_knots=self.edge_knots_,
                spline_order=self.spline_order,
                n_splines=self.n_splines,
                sparse=True,
                periodic=self.basis in ["cp"],
                verbose=verbose,
            )

            edges = np.cumsum([0] + list(counts))
            spline_values = []
            for start, end in zip(edges, edges[1:]):
                spline_values.append(
                    scipy.sparse.csc_matrix(flattened_spline_values[start:end, :].sum(0))
                )
            spline_values = scipy.sparse.vstack(spline_values, format="csc")

            return spline_values
        else:
            spline_values = b_spline_basis(
                X[:, self.feature],
                edge_knots=self.edge_knots_,
                spline_order=self.spline_order,
                n_splines=self.n_splines,
                sparse=True,
                periodic=self.basis in ["cp"],
                verbose=verbose,
            )

        if self.by is not None:
            spline_values = spline_values.multiply(
                np.array(X[:, self.by][:, np.newaxis], dtype="float")
            )

        return spline_values


"""
The below methods introduce patches to PyGAM that are required to enable the
`SharedSpline` term to work properly. Namely, the `SharedSpline` requires `X`
to be an array that contains lists of numbers, which conficts with some of
PyGAM's checks and type casting.
"""


# Need this to circumvent PyGAM checks that would prevent passing data
# containing 'object' type, like lists as in our transformed model data.
class disable_pygam_checks:
    check_X = pygam.pygam.check_X
    check_y = pygam.pygam.check_y
    check_X_y = pygam.pygam.check_X_y

    def __init__(self):
        self.check_X = pygam.pygam.check_X
        self.check_y = pygam.pygam.check_y
        self.check_X_y = pygam.pygam.check_X_y

    def __enter__(self):
        pygam.pygam.check_X = lambda X, *args, **kwargs: X
        pygam.pygam.check_X_y = lambda X, *args, **kwargs: True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pygam.pygam.check_X = self.check_X
        pygam.pygam.check_y = self.check_y
        pygam.pygam.check_X_y = self.check_X_y

    def __call__(self, on=True):
        if on:
            self.__enter__()
        else:
            self.__exit__(None, None, None)


# A hack to fix the following error
# ```TypeError: no supported conversion for types: (dtype('O'),)```
# when calling `build_columns' on a matrix with dtype 'O'.
# Error arises from the fact that the a dtype 'O' is passed to sp.sparse.csc_matrix.

# Solution is to either use a data structure that holds different dtypes for each column (e.g. a pandas DataFrame).
# However, that pandas DataFrame doesn't work since the code base uses __getitem__ in the style of numpy.

# Here, we overwrite `b_spline_basis' in pygam.terms to cast to float.
pygam.terms.b_spline_basis = (
    lambda x, edge_knots, **kwargs: pygam.utils.b_spline_basis(
        np.array(x, dtype="float"), edge_knots, **kwargs
    )
    if x.dtype == "O"
    else pygam.utils.b_spline_basis(x, edge_knots, **kwargs)
)


# Need this patch since newer scipy versions don't coerce the type to float.
pygam.terms.LinearTerm.build_columns = lambda self, X, verbose=False: scipy.sparse.csc_matrix(
    X[:, self.feature][:, np.newaxis].astype(float)
)
