# insurance-fairness-ot

Discrimination-free insurance pricing via Lindholm marginalisation, causal path decomposition, and Wasserstein barycenter correction.

## The problem

UK insurers face a live regulatory obligation, not a theoretical one. The FCA Consumer Duty (PRIN 2A, live July 2023), Equality Act 2010 Section 19, and ICOBS 6B together require demonstrating that pricing models do not systematically disadvantage customers with protected characteristics. The key word is *demonstrate* — annual board attestation, documented methodology, sub-group monitoring.

The hard part is that the regulatory standard is **conditional fairness** (equal price for equal risk), not demographic parity. Young drivers genuinely have more accidents; equalising their premium distribution with older drivers would be actuarially wrong, not fair. Most fairness tooling — including the nearest Python library, EquiPy — targets demographic parity and would over-correct your model.

The correct mathematical framework comes from Lindholm, Richman, Tsanakas and Wüthrich (2022): the discrimination-free price is a marginalisation of the model over the *unconditional* distribution of the protected attribute, equivalent to the causal do-operator. This library implements that, plus the causal path decomposition from Côté, Genest and Abdallah (2025) to separate direct discrimination, proxy discrimination, and actuarially justified effects.

## What it solves that EquiPy doesn't

| Requirement | EquiPy | this library |
|---|---|---|
| Correct fairness criterion (conditional) | No (demographic parity) | Yes (Lindholm) |
| Exposure weighting | No | Yes |
| Causal graph — direct/proxy/justified decomposition | No | Yes |
| GLM-compatible relativity output | No | Yes |
| Frequency/severity decomposition | No | Yes |
| Portfolio bias correction (3 methods) | Implicit | Explicit |
| UK regulatory output (FCA format) | No | Yes |
| Polars-native | No (pandas) | Yes |

## Install

```bash
pip install insurance-fairness-ot
```

Dependencies: numpy, scipy, statsmodels, networkx, POT (Python Optimal Transport), polars.

## Quickstart

```python
import polars as pl
import numpy as np
from insurance_fairness_ot import (
    CausalGraph,
    DiscriminationFreePrice,
    FairnessReport,
    FCAReport,
)

# 1. Specify the causal structure of your pricing model
graph = (CausalGraph()
    .add_protected("gender")
    .add_justified_mediator("claims_history", parents=["gender"])
    .add_proxy("annual_mileage", parents=["gender"])
    .add_outcome("claim_freq")
    .add_edge("claims_history", "claim_freq")
    .add_edge("annual_mileage", "claim_freq"))

# 2. Your trained model (must include gender in training)
def my_model(df: pl.DataFrame) -> np.ndarray:
    # e.g. catboost_model.predict(df) or glm.predict(df)
    ...

# 3. Fit the corrector on calibration data
X_calib = pl.read_parquet("calibration_features.parquet")
D_calib = X_calib.select(["gender"])
exposure_calib = X_calib["exposure"].to_numpy()

dfp = DiscriminationFreePrice(
    graph=graph,
    combined_model_fn=my_model,
    correction="lindholm",        # primary: conditional fairness
    bias_correction="proportional",
)
dfp.fit(X_calib, D_calib, exposure=exposure_calib)

# 4. Apply to new business
X_new = pl.read_parquet("new_business.parquet")
D_new = X_new.select(["gender"])
result = dfp.transform(X_new, D_new)

print(result.fair_premium)         # discrimination-free premium
print(result.bias_correction_factor)  # should be close to 1.0

# 5. FCA compliance report
report = FCAReport(
    result,
    report_metadata={
        "firm_name": "Acme Insurance",
        "model_name": "Motor Frequency GLM v3",
        "reporting_date": "2026-03-10",
        "model_version": "3.0",
    }
)
report.save("fca_fair_value_assessment.md", format="markdown")
report.save("fca_fair_value_assessment.json", format="json")
```

## The math

**Lindholm marginalisation** (primary correction):

```
h*(x_i) = sum_d mu_hat(x_i, d) * P(D=d)
```

For each policyholder, predict what the model would output if they were in each protected group, then average weighted by portfolio proportions. This breaks the correlation between X and D, removing both direct and proxy discrimination while preserving actuarially justified effects.

**Portfolio bias correction**: marginalisation introduces a small bias. Three options:

- `proportional` (default): multiply all fair premiums by `E[Y] / E[h*(X)]` — preserves relativity ordering, compatible with GLM tables
- `uniform`: additive shift
- `kl`: KL-optimal reweighting of `P*(D=d)` — maximum entropy approach

**Wasserstein barycenter** (secondary, for multi-attribute simultaneous correction):

```
m*(x_i) = Q_bar(F_{d_i}(mu_hat(x_i)))
```

where `Q_bar` is the weighted average of per-group quantile functions. Achieves demographic parity. Use after Lindholm for multi-attribute cases.

## Causal graph

The graph classifies variables into four roles:

- **Protected (S)**: gender, disability, ethnicity — must be removed from pricing effect
- **Proxy (V)**: variables that proxy S with no independent causal justification — postcode in some applications, vehicle colour as age proxy
- **Justified mediator (R)**: variables caused by or correlated with S but actuarially legitimate — claims history, NCB years
- **Outcome (Y)**: claims frequency × severity

The Lindholm marginalisation handles all three paths correctly without you needing to manually intervene on them.

## Frequency/severity split

```python
dfp = DiscriminationFreePrice(
    graph=graph,
    frequency_model_fn=freq_model,
    severity_model_fn=sev_model,
    correction="lindholm",
)
result = dfp.fit_transform(X, D, exposure=exposure, y_freq=observed_freq)
# result.freq_fair and result.sev_fair are available separately
```

## GLM relativities

If your downstream system expects multiplicative rating factors, not flat premiums:

```python
corrector = LindholmCorrector(["gender"])
corrector.fit(my_model, X_calib, D_calib)

base_profile = {"vehicle_group": 3, "age_band": "35-44", "ncb": 5, "gender": "F"}
relativities = corrector.get_relativities(my_model, X_new, D_new, base_profile)
# Load these into your GLM parameter table
```

## FCA report output

`FCAReport.render()` produces nine sections covering PS21/11, EP25/2, and Consumer Duty:

1. Executive summary with discrimination metrics before/after
2. Protected characteristics assessed with portfolio shares
3. Methodology explanation in plain English
4. Premium impact by group
5. Causal path attribution
6. Bias correction documentation
7. Limitations and governance notes
8. Equality Act proportionality analysis (template text)
9. Consumer Duty fair value assessment

Available in markdown, JSON, and HTML.

## D paradox

The Lindholm formula requires your model to have been trained with the protected attribute as a feature — you need to predict `mu_hat(x, d)` for all values of `d`. This is intentional: including `d` in training maximises predictive accuracy (the "corrective" fairness family), and marginalisation at prediction time removes the discriminatory effect.

If you cannot collect a protected attribute (common for ethnicity in UK insurance), you must impute `P(D|X)` from external data (e.g. census postcode distributions). This library flags the gap in the FCA report but does not yet implement the imputation.

## Known test values (Lindholm 2022, Example 8)

On the synthetic gender/smoking health insurance example:

- `h*(smoker) = 0.200` — weighted average of 0.2406 (women smoker rate) × 0.4482 + 0.1667 (men smoker rate) × 0.5518
- `h*(non-smoker) = 0.184`
- Portfolio bias = 110.77/112.0 = 0.989
- Proportional correction factor = 1.011

These are implemented as regression tests in `tests/test_correction.py`.

## References

- Lindholm, Richman, Tsanakas, Wüthrich (2022). *Discrimination-Free Insurance Pricing*. ASTIN Bulletin 52(1), 55–89.
- Côté, Genest, Abdallah (2025). *A fair price to pay: Exploiting causal graphs for fairness in insurance*. Journal of Risk and Insurance 92(1), 33–75.
- Charpentier, Hu, Ratz (2023). *Mitigating Discrimination in Insurance with Wasserstein Barycenters*. arXiv:2306.12912.

## Performance

No formal benchmark yet. The Lindholm marginalisation requires one model prediction per protected group value per policyholder. For a binary protected characteristic (e.g. gender), this doubles the number of predictions. For a 5-category characteristic (e.g. age band used as a protected attribute), it is 5x. On a portfolio of 100,000 policies with 2 protected groups, this takes 1–5 seconds with a typical CatBoost model.

The Wasserstein barycenter correction (correction='wasserstein') adds optimal transport computation via POT. On n=10,000 calibration samples with a 5-category protected attribute, expect 10–60 seconds depending on the regularisation parameter. For production use with large portfolios, fit the corrector on a calibration subsample (default behaviour) and apply to the full book in one pass.

The portfolio bias correction factor should be close to 1.0 (within ±5%) for well-calibrated models. Values outside this range indicate the model has substantial discrimination baked into its calibration, not just its predictions. This is the primary diagnostic: if the proportional correction factor is 1.15, the model is overcharging the disadvantaged group by 15% on average after marginalisation.

## Licence

MIT
