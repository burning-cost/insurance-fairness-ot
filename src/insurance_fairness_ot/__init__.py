"""insurance-fairness-ot: Discrimination-free insurance pricing.

Implements Lindholm (2022) marginalisation, Côté (2025) causal path
decomposition, and Wasserstein barycenter correction for UK personal lines.

Quickstart::

    from insurance_fairness_ot import (
        CausalGraph,
        DiscriminationFreePrice,
        FairnessReport,
        FCAReport,
    )

    graph = (CausalGraph()
        .add_protected("gender")
        .add_justified_mediator("claims_history", parents=["gender"])
        .add_proxy("annual_mileage", parents=["gender"])
        .add_outcome("claim_freq"))

    dfp = DiscriminationFreePrice(graph=graph, combined_model_fn=my_model)
    result = dfp.fit_transform(X_train, D_train, exposure=exposure_train)
"""

from .causal import CausalGraph, PathDecomposer, PathDecomposition
from .correction import LindholmCorrector, WassersteinCorrector
from .pricing import DiscriminationFreePrice, PricingResult
from .report import FairnessReport, FCAReport

__all__ = [
    "CausalGraph",
    "PathDecomposer",
    "PathDecomposition",
    "LindholmCorrector",
    "WassersteinCorrector",
    "DiscriminationFreePrice",
    "PricingResult",
    "FairnessReport",
    "FCAReport",
]

__version__ = "0.1.0"
