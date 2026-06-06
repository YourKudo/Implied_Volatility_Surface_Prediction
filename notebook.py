"""
NIFTY Options IV Surface Reconstruction
Algorithm: Cross-sectional polynomial smile fit + Dual-EWMA residual correction
Strictly causal (no look-ahead bias).
"""
import numpy as np
import pandas as pd

# ---------- Hyperparameters ----------
ALPHA_FAST = 0.30   # fast EWMA decay
ALPHA_SLOW = 0.05   # slow EWMA decay
W_FAST     = 0.60   # weight on fast EWMA
BASE_CAP   = 0.015  # max absolute correction
CAP_FRAC   = 0.10   # cap as a fraction of predicted IV
MIN_CAP    = 0.005  # floor on the adaptive cap

# ---------- Load ----------
df = pd.read_csv("dataset.csv")
iv_cols = [c for c in df.columns if c not in ("datetime", "underlying_price")]
ce_cols = [c for c in iv_cols if c.endswith("CE")]
pe_cols = [c for c in iv_cols if c.endswith("PE")]
print(f"Loaded: {len(df)} rows, {len(iv_cols)} IV cols ({len(ce_cols)} CE, {len(pe_cols)} PE)")
print(f"Missing cells to fill: {df[iv_cols].isna().sum().sum()}")


# ---------- Cross-sectional smile prediction ----------
def cs_predict(row, col, group, is_ce):
    """
    Predict IV at strike k0 using the volatility smile of `group`
    (CE strikes or PE strikes) at the *same* timestamp.
    Uses inverse-distance weighted polynomial fit, with sanity fallback.
    """
    k0 = int(col[12:-2])
    obs = sorted(
        [(int(c[12:-2]), float(row[c])) for c in group
         if c != col and pd.notna(row[c]) and float(row[c]) > 0],
        key=lambda x: x[0]
    )
    if len(obs) < 2:
        return np.nan

    bl = sorted([(k, v) for k, v in obs if k < k0], key=lambda x: -x[0])  # below, closest first
    ab = sorted([(k, v) for k, v in obs if k > k0], key=lambda x:  x[0])  # above, closest first

    # Two-sided: weighted polynomial smile fit
    if bl and ab:
        pts = bl[:4] + ab[:4]
        pts.sort(key=lambda x: x[0])
        sk = np.array([p[0] for p in pts])
        sv = np.array([p[1] for p in pts])
        try:
            # Inverse-distance weights, closer strikes dominate
            dists = np.abs(sk - k0).astype(float)
            dists[dists < 50] = 50  # avoid div-by-zero
            weights = 1.0 / dists
            cf = np.polyfit(sk, sv, min(2, len(sk) - 1), w=weights)
            pred = float(np.polyval(cf, k0))

            # Sanity: prediction should be within reasonable range of neighbors
            local_min = min(sv[np.argmin(np.abs(sk - k0))], sv.min())
            local_max = max(sv[np.argmax(np.abs(sk - k0))], sv.max())
            if pred < local_min * 0.5 or pred > local_max * 2.0:
                # Fall back to linear interpolation between nearest neighbors
                lK, lIV = bl[0]; rK, rIV = ab[0]
                return lIV + (k0 - lK) / (rK - lK) * (rIV - lIV)
            return pred
        except Exception:
            lK, lIV = bl[0]; rK, rIV = ab[0]
            return lIV + (k0 - lK) / (rK - lK) * (rIV - lIV)

    # Boundary / one-sided extrapolation
    side_all = sorted(bl if bl else ab, key=lambda x: abs(x[0] - k0))
    obs_ks = [s[0] for s in side_all]
    going_otm = (k0 > max(obs_ks)) if is_ce else (k0 < min(obs_ks))
    side = side_all[:3] if going_otm else side_all[:2]
    side.sort(key=lambda x: x[0])
    sk = [p[0] for p in side]; sv = [p[1] for p in side]
    try:
        return float(np.polyval(np.polyfit(sk, sv, min(1, len(sk) - 1)), k0))
    except Exception:
        return side[0][1]


# ---------- PASS 1: build per-strike CS residual table (causal) ----------
print("\nPass 1: computing cross-section residuals for all observed cells ...")
residuals = {col: {} for col in iv_cols}

for i in range(len(df)):
    row = df.iloc[i]
    for col in iv_cols:
        if pd.notna(row[col]):
            is_ce = col.endswith("CE")
            group = ce_cols if is_ce else pe_cols
            p = cs_predict(row, col, group, is_ce)
            if pd.notna(p) and p > 0.005:
                residuals[col][i] = float(row[col]) - p
    if (i + 1) % 250 == 0:
        print(f"  row {i+1}/{len(df)}")
print(f"  residuals stored: {sum(len(v) for v in residuals.values())}")
# NOTE: outlier clipping on residuals was REMOVED.
#       The original col_resid_stats computed mean/std from ALL timestamps
#       including future ones -- a subtle future-data leak.
#       The EWMA cap already provides sufficient protection.


# ---------- DUAL-EWMA CORRECTION (strictly causal, adaptive cap) ----------
def get_correction(col, before_row, predicted_iv=None):
    past = sorted([(idx, r) for idx, r in residuals[col].items() if idx < before_row])
    if not past:
        return 0.0

    # EWMA on raw residuals (no clipping needed -- cap handles outliers)
    ef = es = past[0][1]
    for _, r in past[1:]:
        ef = ALPHA_FAST * r + (1 - ALPHA_FAST) * ef
        es = ALPHA_SLOW * r + (1 - ALPHA_SLOW) * es
    ewma = W_FAST * ef + (1 - W_FAST) * es

    # Adaptive cap (uses only current-timestamp cross-section prediction)
    if predicted_iv is not None and predicted_iv > 0.05:
        cap = min(BASE_CAP, CAP_FRAC * abs(predicted_iv))
        cap = max(cap, MIN_CAP)
    else:
        cap = BASE_CAP
    return float(np.clip(ewma, -cap, cap))


# ---------- PASS 2: fill missing values ----------
print("\nPass 2: filling missing values with CS + EWMA correction ...")
filled = df.copy()
fill_count = 0
for i in range(len(df)):
    row = df.iloc[i]
    for col in iv_cols:
        if pd.isna(row[col]):
            is_ce = col.endswith("CE")
            group = ce_cols if is_ce else pe_cols
            p = cs_predict(row, col, group, is_ce)
            if pd.isna(p):
                # last-resort: column running mean of past observed values
                past_vals = [df.iloc[j][col] for j in range(i) if pd.notna(df.iloc[j][col])]
                p = float(np.mean(past_vals)) if past_vals else 0.15
            corr = get_correction(col, i, predicted_iv=p)
            filled.at[i, col] = max(0.005, p + corr)
            fill_count += 1
    if (i + 1) % 250 == 0:
        print(f"  row {i+1}/{len(df)}")
print(f"  filled cells: {fill_count}")

# Final sanity check
remaining = filled[iv_cols].isna().sum().sum()
print(f"  remaining NaN: {remaining}")

filled.to_csv("filled_dataset.csv", index=False)
print("\nSaved filled_dataset.csv")




# ---------- CREATE SUBMISSION FILE ----------
SEPARATOR = "||"

rows = []

for col in iv_cols:

    # Find locations that were originally missing
    missing_mask = df[col].isna()

    for idx in df.index[missing_mask]:

        rows.append({
            "id": (
                f"{df.loc[idx, 'datetime']}"
                f"{SEPARATOR}"
                f"{col}"
            ),
            "value": float(filled.loc[idx, col])
        })

submission = (
    pd.DataFrame(rows)
    .sort_values("id")
    .reset_index(drop=True)
)

submission.to_csv(
    "submission.csv",
    index=False
)

print(f"Saved submission.csv ({len(submission)} rows)")