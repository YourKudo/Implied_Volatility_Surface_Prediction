# NIFTY Implied-Volatility Surface Imputation

> Finance Club, IIT Roorkee — **Open Project 2026** submission.

Reconstruct the missing entries of a partially-observed implied-volatility (IV) grid for NIFTY index options at a single expiry. The pipeline is two-stage: a same-timestamp volatility-smile estimator gives the bulk of each prediction, and a gradient-boosted residual model corrects what the smile misses.

---

## Problem

We are given a 5-minute snapshot grid of NIFTY option implied volatilities across 21 trading days (07–27 Jan 2026) at a single expiry (27JAN26). About 20% of the IV cells are missing — these are the cells we are scored on (mean squared error).

- **975 timestamps** (rows), every 5 minutes across the trading window
- **28 IV columns**: 14 calls (`...CE`) and 14 puts (`...PE`)
- **One expiry only** ⇒ no term structure, just a single smile evolving in time
- **Disjoint strike support** for the call wing (25200–26500) and put wing (23800–25100) — they are not interchangeable views of the same point
- **Liquidity-driven missingness**: OTM strikes drop out far more often than ATM, so cross-strike interpolation from neighbours of the *same type* is the strongest signal available

The submission is a CSV of `(id, value)` where `id = "datetime||ticker"` and `value` is the predicted IV for that missing cell.

---

## Approach in one picture

$$
\widehat{\sigma}(t, K) \;=\; \underbrace{\hat{\sigma}_{\text{smile}}(t, K)}_{\text{Stage 1: IDW-weighted quadratic smile fit}}
\;+\;
\underbrace{\bar{r}_{\text{boost}}(t, K)}_{\text{Stage 2: averaged HGBM residual}}
$$

Stage 1 explains most of the cross-strike variation cheaply and robustly. Stage 2 learns the small structured residual that the smile misses (intraday drift, strike-specific quote noise, wing distortion). Training the booster on residuals — not on raw IV — is what makes the model both stable and accurate: it never has to relearn the smile, only its corrections.

---

## Algorithm

### Stage 1 · Cross-sectional smile estimator

For each cell `(t, K, type)`, look at the same-timestamp smile of the same option type (CE or PE):

1. Collect observed `(strike, IV)` pairs from the same row, same type, excluding the target.
2. Split into points **below** and **above** the target strike $K$.
3. **Two-sided fit** (the common case): take up to four neighbours below and four above, fit a polynomial of degree `min(2, n − 1)` with inverse-distance weights

$$
w_i = \frac{1}{\max(|K_i - K|,\,50)}
$$

   The 50-point floor stops the closest neighbour from dominating numerically.

4. **Sanity bound**: if the quadratic estimate falls outside `[0.5 · local_min, 2 · local_max]` of the support points, discard it and fall back to a linear blend between the single nearest below and single nearest above.
5. **One-sided wings**: if only one side has observations, use a low-order fit over the 2–3 nearest same-side points (3 if extrapolating beyond the observed chain, 2 if interpolating inside it). Fall back to nearest-neighbour value on numerical failure.

The whole estimator is **same-timestamp only**, with no temporal lookup and no future leakage.

### Stage 2 · Residual learner

The boosted model is trained to predict $r = \sigma_{\text{actual}} - \hat{\sigma}_{\text{smile}}$ from a 13-dimensional feature vector. At inference we add the residual back: $\hat{\sigma} = \hat{\sigma}_{\text{smile}} + \hat{r}$.

Two estimators are trained on identical data and averaged:

| Model | Configuration |
|---|---|
| `HistGradientBoostingRegressor` | `max_iter=400`, `learning_rate=0.05`, `l2_regularization=1.0`, `min_samples_leaf=20`, `random_state=0` |
| `BaggingRegressor` over the above | `n_estimators=10`, `max_samples=0.8`, `max_features=0.8`, `random_state=0` |

The single HGBM is sharper; the bagged HGBM is smoother and more robust at the wings where data is thin. Averaging recovers most of the sharpness while damping the wing variance — neither alone matched the average on the public leaderboard.

**Strict training rule:** observed cells that are not the targets for this run. Missing cells (and held-out cells, during validation) never enter training.

### Feature matrix

Every cell — observed or missing — gets a 13-feature row. All features are same-timestamp or **strictly past** relative to the cell; nothing uses a future value of the cell itself.

| # | Feature | Captures |
|---|---|---|
| 1 | $\log(K / S)$ | log-moneyness |
| 2 | $K$ | raw strike |
| 3 | `is_call` | 1 for CE, 0 for PE |
| 4 | $S$ | spot at this timestamp |
| 5 | $t / T$ | normalised time index |
| 6 | $\operatorname{median}(\text{obs IVs at } t)$ | smile level |
| 7 | $\operatorname{std}(\text{obs IVs at } t)$ | smile dispersion |
| 8 | $\hat{\sigma}_{\text{smile}}$ | the Stage-1 estimate itself |
| 9 | last observed IV in same column | temporal carry-forward |
| 10–11 | nearest *lower* neighbour: IV, $K - K_{\text{lo}}$ | local left context |
| 12–13 | nearest *upper* neighbour: IV, $K_{\text{hi}} - K$ | local right context |

Including the smile estimate (feature 8) as an input makes the residual learning robust: when the smile fit is unreliable (e.g. wings with few neighbours), features 6–7 and 10–13 give the booster enough context to override it.

---

## Validation

`holdout_validate` hides a random 15% of the **observed** cells, runs the full pipeline against the masked frame, and scores against the true values that were hidden. Held-out cells are excluded from training (no leakage).

**Important caveat noted up-front**: the cells we can hide are the *liquid* ones, whereas the leaderboard scores the *illiquid* (truly missing) ones. So this metric tends to *under-estimate* test MSE for any high-capacity model. The self-validation number was used as a sanity check; the public leaderboard was the actual selection metric.

```
Self-validation MSE (3262 held-out liquid cells, seed 0): 0.000034
```

---

## Reproducing

Requirements: `numpy`, `pandas`, `scikit-learn`.

```bash
pip install numpy pandas scikit-learn
jupyter notebook IV_imputation_restructured.ipynb
```

Or, headless:

```bash
jupyter nbconvert --to notebook --execute IV_imputation_restructured.ipynb
```

Running top-to-bottom produces:

- `filled_dataset.csv` — the full grid with the 5460 missing cells imputed
- `submission.csv` — the 5460-row Kaggle submission file with `id, value` columns

Every random component is seeded (`SEED = 0`); re-runs are bit-identical.

---

## Repository layout

```
.
├── IV_imputation_restructured.ipynb   # main notebook (run this)
├── dataset.csv                         # input grid (provided by the competition)
├── filled_dataset.csv                  # output: full grid with imputed cells
├── submission.csv                      # output: Kaggle submission
└── README.md
```

---

## Design notes & what didn't make the cut

A few directions tried and rejected during development:

- **Per-type pooling vs. cross-type pooling.** Pooling CE and PE into a single smile fit reliably *hurt*; CE and PE are kept on separate smiles throughout. The disjoint strike support means there are no shared anchor points, and forcing a joint fit warps both wings.
- **Strictly causal temporal models** (EWMA, Holt-Winters on residuals). Effective on their own, but the dataset is fully observed at imputation time — there's no information-theoretic reason to enforce causality. The HGBM uses both same-row context and a strictly-past carry-forward, which empirically dominated causal-only residual models.
- **Higher polynomial degree** in the smile fit. Cubic and quartic overshoot the wings on thin support; the quadratic plus the sanity-band fallback was both more accurate and more stable.
- **Iterative imputation** (`IterativeImputer` with a Ridge / BayesianRidge backbone). Lower in-sample error but did not transfer — the predicted IVs at the illiquid strikes drifted toward the column means and lost the smile shape.

---

## Acknowledgements

NIFTY options data from the Finance Club, IIT Roorkee — Open Project 2026 dataset.
