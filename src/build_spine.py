#!/usr/bin/env python3
"""
SMB Underwriting Challenge -- MODELING SPINE (Prompt 1 of 2).

Shared discrete-time time-to-default hazard model consumed by Deliverables A and B,
plus a self-validation back-test harness. Writes artifacts only; NO submission files.

GBM note: LightGBM/XGBoost both require an OpenMP runtime (libomp) that is not
installable in this sandbox (no Homebrew). We use sklearn HistGradientBoostingClassifier
-- a histogram gradient-boosting model that (a) is a true GBM, (b) handles NaN natively
(critical for the MNAR bank-feed block and informative nulls), (c) supports monotonic /
categorical constraints, (d) needs no OpenMP. Functionally equivalent for this spine.

Discrete-time survival references (for the writeup, Deliverable D):
  * Singer & Willett (2003), Applied Longitudinal Data Analysis, ch. 10-12
    (person-period reshape; discrete-time hazard via pooled logistic).
  * Efron (1988, JASA 83:414) -- logistic regression as discrete-survival hazard.
  * Prentice & Gloeckler (1978, Biometrics 34:57) -- grouped/discrete survival hazard.
Covariate-shift transport references:
  * Shimodaira (2000, J.Stat.Plan.Inf. 90:227) -- importance-weighted ERM under shift.
  * Sugiyama et al. (2008, AISM 60:699) -- KLIEP direct density-ratio estimation.
We use a probabilistic train-vs-target discriminator for the density ratio (equivalent
class-probability route to the same w(x)=p_target/p_source).
"""
import os, json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
os.makedirs("artifacts", exist_ok=True)

# ----------------------------------------------------------------------------- #
# Column contracts
# ----------------------------------------------------------------------------- #
ID = ["business_id", "applicant_id"]
TIME = "application_timestamp"
LEAK = ["days_to_default", "days_to_full_repayment", "repayment_status",
        "final_recovered_amount", "observation_status"]          # post-outcome leak
COLLIDER = ["prior_decision", "prior_approved_amount"]            # selection colliders
LABEL = "default_flag"
CAT = ["sector", "geography_region", "employee_count_bucket",
       "intended_use_of_funds", "owner_personal_credit_band", "application_channel"]
OPAQUE = "requested_amount_to_observed_revenue"
BANKFEED = ["observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
            "observed_revenue_volatility", "observed_cash_balance_p10",
            "observed_overdraft_count_3mo", "payroll_regularity_score"]

def wk(days):                      # day -> age-week, 1..13
    return np.clip(np.ceil(np.asarray(days, float) / 7.0), 1, 13).astype(int)

def build_features(df, include_opaque, onehot_cols=None):
    """Leak/collider-free feature matrix with explicit missingness signal."""
    X = pd.DataFrame(index=df.index)
    # explicit missingness indicators (audit: these ARE the signal)
    X["miss_ext_decline"]   = df["days_since_last_external_decline"].isna().astype(int)
    X["miss_inq_elsewhere"] = df["days_since_last_inquiry_elsewhere"].isna().astype(int)
    X["no_bank_feed"]       = (~df["has_linked_bank_feed"].astype(bool)).astype(int)
    X["has_linked_bank_feed"] = df["has_linked_bank_feed"].astype(int)
    # numerics kept as-is (HGB routes NaN natively -> no destructive imputation)
    num = ["vintage_years", "stated_annual_revenue", "stated_time_in_business",
           "requested_amount", *BANKFEED, "aggregate_credit_utilization",
           "recent_inquiries_count_6mo", "existing_debt_obligations",
           "days_since_last_external_decline", "account_age_days",
           "platform_active_months", "bookkeeping_recency_days",
           "invoice_payment_delinquency_rate", "prior_loans_count",
           "prior_loans_default_count", "prior_loans_amount_total",
           "multi_lender_inquiry_count_30d", "days_since_last_inquiry_elsewhere",
           "repeat_application_count"]
    # prior_underwriter_score DROPPED (Prompt-2 STEP 0 ablation): keeping it changes
    # held-out cal log-loss by only -0.0002 (< 0.001 bar) AND it is a prior-model output
    # -> a selection-bias vector. Dropping it is marginally better and causally cleaner.
    for c in num:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    if include_opaque:
        X[OPAQUE] = pd.to_numeric(df[OPAQUE], errors="coerce")
    # one-hot categoricals (all levels appear in train -> stable columns)
    d = pd.get_dummies(df[CAT].astype("Int64").astype("category"), prefix=CAT, dummy_na=False)
    X = pd.concat([X, d.astype(int)], axis=1)
    if onehot_cols is not None:                 # align to training columns
        for c in onehot_cols:
            if c not in X.columns: X[c] = 0
        X = X[onehot_cols]
    return X

def hgb(**kw):
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
             min_samples_leaf=80, l2_regularization=1.0,
             early_stopping=True, validation_fraction=0.12,
             random_state=RNG)
    p.update(kw); return HistGradientBoostingClassifier(**p)

# ----------------------------------------------------------------------------- #
# Load
# ----------------------------------------------------------------------------- #
tr = pd.read_csv("dataset/train.csv")
va = pd.read_csv("dataset/validation.csv")
te = pd.read_csv("dataset/test.csv")
tr[TIME] = pd.to_datetime(tr[TIME]); va[TIME] = pd.to_datetime(va[TIME]); te[TIME] = pd.to_datetime(te[TIME])

lab = tr[tr[LABEL].notna()].copy().sort_values(TIME).reset_index(drop=True)   # 51,722 matured
lab["y"] = lab[LABEL].astype(int)
lab["event_week"] = np.where(lab["y"] == 1, wk(lab["days_to_default"]), 0)
# phase split: missed-draw (weeks 1..9) vs maturity ATOM (event_week>=10, empirically all 13)
PH1_MAX = 9
lab["phase1_default"] = ((lab["y"] == 1) & (lab["event_week"] <= PH1_MAX)).astype(int)
lab["atom_default"]   = ((lab["y"] == 1) & (lab["event_week"] >= 10)).astype(int)
print(f"[load] labeled={len(lab)}  default_rate={lab['y'].mean():.4f}  "
      f"phase1_def={lab['phase1_default'].sum()}  atom_def={lab['atom_default'].sum()}  "
      f"(atom share of defaults={lab['atom_default'].sum()/lab['y'].sum():.3f})")

# ----------------------------------------------------------------------------- #
# Temporal folds (respect time ordering): fit 60% | calib 20% | backtest 20%
# ----------------------------------------------------------------------------- #
n = len(lab); i1, i2 = int(0.60 * n), int(0.80 * n)
lab["fold"] = np.where(np.arange(n) < i1, "fit", np.where(np.arange(n) < i2, "cal", "back"))
folds = {k: lab[lab.fold == k].copy() for k in ["fit", "cal", "back"]}
for k, g in folds.items():
    print(f"[fold] {k:4s} n={len(g):6d}  {g[TIME].min().date()}..{g[TIME].max().date()}  dr={g['y'].mean():.4f}")

# ----------------------------------------------------------------------------- #
# Person-period reshape (weeks 1..9) for the phase-1 hazard
# ----------------------------------------------------------------------------- #
def person_period(sub, Xsub):
    """One row per loan per at-risk week k in 1..9. event = default in week k."""
    rows_idx, weeks, ev = [], [], []
    last = np.where(sub["phase1_default"] == 1, sub["event_week"], PH1_MAX).astype(int)  # at-risk through `last`
    is_p1 = sub["phase1_default"].values
    ew = sub["event_week"].values
    for pos, (lastk, p1, e) in enumerate(zip(last, is_p1, ew)):
        for k in range(1, lastk + 1):
            rows_idx.append(pos); weeks.append(k); ev.append(1 if (p1 and k == e) else 0)
    PP = Xsub.iloc[rows_idx].reset_index(drop=True)
    PP["age_week"] = weeks
    return PP, np.array(ev)

def fit_models(include_opaque):
    Xfit = build_features(folds["fit"], include_opaque)
    onehot = list(Xfit.columns)
    PPx, PPy = person_period(folds["fit"].reset_index(drop=True), Xfit.reset_index(drop=True))
    h1 = hgb().fit(PPx, PPy)
    # atom head: among survivors of phase1 (non-phase1-default), predict maturity default
    surv = folds["fit"]["phase1_default"] == 0
    Xa = Xfit[surv.values]; ya = folds["fit"].loc[surv, "atom_default"].values
    atom = hgb(max_iter=300).fit(Xa, ya)
    return dict(h1=h1, atom=atom, onehot=onehot, include_opaque=include_opaque)

def predict_curves(M, df):
    """Return h[1..13] and F[1..13] (cumulative incidence) per row; lifetime PD = F13."""
    X = build_features(df, M["include_opaque"], onehot_cols=M["onehot"])
    N = len(X)
    H = np.zeros((N, 14))                       # 1-indexed weeks
    for k in range(1, PH1_MAX + 1):
        Xk = X.copy(); Xk["age_week"] = k
        H[:, k] = M["h1"].predict_proba(Xk)[:, 1]
    # weeks 10,11,12 := 0 (structural zeros, enforced)
    a13 = M["atom"].predict_proba(X)[:, 1]      # P(atom default | survived to maturity)
    # cumulative incidence
    S = np.ones(N); F = np.zeros((N, 14))
    for k in range(1, PH1_MAX + 1):
        F[:, k] = F[:, k - 1] + S * H[:, k]; S = S * (1 - H[:, k])
    for k in (10, 11, 12):
        F[:, k] = F[:, k - 1]                   # flat (h=0)
    H[:, 13] = a13
    F[:, 13] = F[:, 12] + S * a13               # S here = S(9)=S(12)
    return H[:, 1:14], F[:, 1:14]               # shape (N,13)

# ----------------------------------------------------------------------------- #
# Feature toggle: with vs without the opaque engineered ratio (pick by cal log-loss)
# ----------------------------------------------------------------------------- #
def cal_logloss(M):
    _, F = predict_curves(M, folds["cal"])
    pd_life = F[:, 12]
    return log_loss(folds["cal"]["y"].values, np.clip(pd_life, 1e-6, 1 - 1e-6))

M_with = fit_models(True); ll_with = cal_logloss(M_with)
M_without = fit_models(False); ll_wo = cal_logloss(M_without)
USE_OPAQUE = ll_with < ll_wo
M = M_with if USE_OPAQUE else M_without
print(f"[feat] cal log-loss with_opaque={ll_with:.5f}  without={ll_wo:.5f}  -> USE_OPAQUE={USE_OPAQUE}")

# ----------------------------------------------------------------------------- #
# In-regime isotonic calibration (fit on CAL fold) -- used for the back-test
# ----------------------------------------------------------------------------- #
_, Fcal = predict_curves(M, folds["cal"])
iso_inreg = IsotonicRegression(out_of_bounds="clip").fit(Fcal[:, 12], folds["cal"]["y"].values)

# ----------------------------------------------------------------------------- #
# STEP 3 -- REGIME TRANSPORT (two methods)
# ----------------------------------------------------------------------------- #
# Method 1: density-ratio w(x)=p_target/p_source via train-vs-target discriminator,
# cross-fitted. Source = labeled train (fit+cal). Target = val+test full population.
src = pd.concat([folds["fit"], folds["cal"]]).reset_index(drop=True)
tgt = pd.concat([va, te]).reset_index(drop=True)
Xsrc = build_features(src, M["include_opaque"], onehot_cols=M["onehot"])
Xtgt = build_features(tgt, M["include_opaque"], onehot_cols=M["onehot"])
Dx = pd.concat([Xsrc, Xtgt], ignore_index=True)
Dy = np.r_[np.zeros(len(Xsrc)), np.ones(len(Xtgt))]
oof = np.zeros(len(Dx)); idx = np.arange(len(Dx)); np.random.shuffle(idx)
half = len(idx) // 2; A, B = idx[:half], idx[half:]
for tr_i, pr_i in [(A, B), (B, A)]:
    disc = hgb(max_iter=250).fit(Dx.iloc[tr_i], Dy[tr_i])
    oof[pr_i] = disc.predict_proba(Dx.iloc[pr_i])[:, 1]
d_src = np.clip(oof[:len(Xsrc)], 1e-4, 1 - 1e-4)
w = (d_src / (1 - d_src)) * (len(Xsrc) / len(Xtgt))      # = p_target/p_source
w = np.clip(w, 0.05, 20.0)
_, Fsrc = predict_curves(M, src)
pd_src_raw = Fsrc[:, 12]
base_method1 = np.average(pd_src_raw, weights=w)          # transported base rate
disc_auc = None
try:
    from sklearn.metrics import roc_auc_score
    disc_auc = roc_auc_score(Dy, oof)
except Exception: pass

# Method 2: recency holdout -- latest 3 months of labeled train as a shift proxy.
cut = lab[TIME].max() - pd.Timedelta(days=90)
rec = lab[lab[TIME] >= cut]
base_method2 = rec["y"].mean()                           # realized dr in recency proxy
_, Frec = predict_curves(M, rec); pd_rec_raw = Frec[:, 12].mean()
# anchor the transported calibration to realized recency labels (validated by val 0.206)
iso_test = IsotonicRegression(out_of_bounds="clip").fit(Frec[:, 12], rec["y"].values)

gap = abs(base_method1 - base_method2)
print(f"[transport] disc_AUC={disc_auc:.3f}  w[min/med/max]={w.min():.2f}/{np.median(w):.2f}/{w.max():.2f}")
print(f"[transport] method1 density-ratio base rate = {base_method1:.4f}")
print(f"[transport] method2 recency-holdout base rate = {base_method2:.4f} (n_recency={len(rec)}, raw pred there={pd_rec_raw:.4f})")
print(f"[transport] RECONCILE gap = {gap:.4f}  (train base=0.1745, val observed=0.206)")

# Final scoring calibration is ANCHORED to the realized recent regime (Method 2),
# which the validation labels independently confirm (val approved dr 0.206 ~ recency 0.211).
# The density-ratio covariate transport (Method 1 = 0.166) points the OTHER way: test
# applicants are bigger/richer, which under TRAIN relationships maps to LOWER default --
# but realized default is HIGHER. So the shift is concept/label (worse macro: revenue
# trend +0.15 -> -0.45), not pure covariate. We trust the realized signal for the LEVEL
# and treat the 0.045 method gap as irreducible regime uncertainty (sizes conformal width
# in Prompt 2). iso_test is fit in the Method-2 block below, once Frec is available.

# ----------------------------------------------------------------------------- #
# STEP 4 -- BACK-TEST HARNESS (held-out 'back' fold, true labels, in-regime calib)
# ----------------------------------------------------------------------------- #
Hb, Fb = predict_curves(M, folds["back"])
yb = folds["back"]["y"].values
pd_back = np.clip(iso_inreg.transform(Fb[:, 12]), 1e-6, 1 - 1e-6)
ll = log_loss(yb, pd_back); br = brier_score_loss(yb, pd_back)

# (a) reliability by decile
dec = pd.qcut(pd_back, 10, labels=False, duplicates="drop")
rel = pd.DataFrame({"decile": dec, "pred": pd_back, "real": yb}).groupby("decile").agg(
    n=("real", "size"), pred=("pred", "mean"), real=("real", "mean")).reset_index()

# (b) realized vs predicted cumulative hazard by age-week (shape reproduction)
#     realized cumulative default rate by age = fraction of fold defaulting by week a
ew_b = folds["back"]["event_week"].values; yb_ = yb
real_cdr = np.array([np.mean((yb_ == 1) & (ew_b <= a)) for a in range(1, 14)])
pred_cdr = Fb.mean(axis=0)                                # model's mean cumulative incidence

# external check on VALIDATION labels (approved subset) with transported calibration
vlab = va[va[LABEL].notna()].copy()
_, Fv = predict_curves(M, vlab)
pd_v = np.clip(iso_test.transform(Fv[:, 12]), 1e-6, 1 - 1e-6)
val_ll = log_loss(vlab[LABEL].astype(int), pd_v); val_mean_pd = pd_v.mean()
val_real = vlab[LABEL].mean()

# ----------------------------------------------------------------------------- #
# OUTPUT ARTIFACTS
# ----------------------------------------------------------------------------- #
pickle.dump({"model": M, "iso_inreg": iso_inreg, "iso_test": iso_test,
             "use_opaque": bool(USE_OPAQUE), "ph1_max": PH1_MAX,
             "transport": {"method1_base": float(base_method1),
                           "method2_base": float(base_method2), "gap": float(gap),
                           "disc_auc": float(disc_auc) if disc_auc else None}},
            open("artifacts/hazard_model.pkl", "wb"))

# per-applicant curves for ALL 13,306 val+test
out = []
for split, df in [("validation", va), ("test", te)]:
    H, F = predict_curves(M, df)
    Fc = np.clip(iso_test.transform(F.reshape(-1)).reshape(F.shape), 0, 1)
    Fc = np.maximum.accumulate(Fc, axis=1)               # keep monotone after calib
    rec_df = pd.DataFrame({"applicant_id": df["applicant_id"].values, "split": split})
    for k in range(13):
        rec_df[f"h{k+1}"] = H[:, k]
    for k in range(13):
        rec_df[f"F{k+1}"] = Fc[:, k]
    rec_df["pd_lifetime_cal"] = Fc[:, 12]
    out.append(rec_df)
curves = pd.concat(out, ignore_index=True)
curves.to_csv("artifacts/applicant_curves.csv", index=False)

pd.DataFrame({"applicant_id": src["applicant_id"].values, "w": w,
              "pd_raw": pd_src_raw}).to_csv("artifacts/density_ratio_weights.csv", index=False)

# back-test report
def fmt_row(r): return f"| {int(r.decile)} | {int(r.n)} | {r.pred:.3f} | {r.real:.3f} | {r.pred-r.real:+.3f} |"
shape_ok = bool(np.all(np.abs(pred_cdr[9:12] - pred_cdr[8]) < 1e-9) and (pred_cdr[12] - pred_cdr[11] > 0.01))
report = f"""# BACKTEST REPORT -- Modeling Spine (Deliverables A+B shared hazard)

**Model:** discrete-time time-to-default hazard. Phase-1 pooled person-period GBM over
age-weeks 1-9 (HistGradientBoosting; LightGBM/XGBoost unavailable -- no libomp in sandbox),
structural zeros at weeks 10-12, separate week-13 maturity-atom head. Cumulative incidence
F(a)=1-S(a) is monotone by construction. Feature set uses the opaque engineered ratio: **{USE_OPAQUE}**
(chosen by held-out cal log-loss: with={ll_with:.5f} vs without={ll_wo:.5f}).

Temporal folds (no random leakage): fit {len(folds['fit'])} | cal {len(folds['cal'])} | back {len(folds['back'])}.

## (a) PD calibration on held-out BACK fold (in-regime isotonic, fit on CAL)
log-loss = **{ll:.5f}**  |  Brier = **{br:.5f}**  |  n = {len(yb)}  |  realized dr = {yb.mean():.4f}  |  mean pred = {pd_back.mean():.4f}

| decile | n | mean predicted PD | realized default rate | gap |
|---|---|---|---|---|
""" + "\n".join(fmt_row(r) for r in rel.itertuples()) + f"""

## (b) Hazard-SHAPE reproduction (mean cumulative default rate by age-week)
Does the model reproduce the flat weeks 10-12 and the day-90 atom?  **reproduced = {shape_ok}**

| age wk | predicted cum | realized cum |
|---|---|---|
""" + "\n".join(f"| {a+1} | {pred_cdr[a]:.4f} | {real_cdr[a]:.4f} |" for a in range(13)) + f"""

Flat-region check: pred F10=F11=F12=F9 ({pred_cdr[8]:.4f}); week-13 jump = {pred_cdr[12]-pred_cdr[11]:+.4f}.

## (c) Aggregate scores
back-fold log-loss {ll:.5f}, Brier {br:.5f}.

## External check -- VALIDATION labels (approved), TRANSPORTED calibration
val log-loss = {val_ll:.5f} | mean predicted PD = **{val_mean_pd:.4f}** | realized (approved) = {val_real:.4f}
(target: pull mean PD toward the ~0.206 val regime, away from the 0.1745 train rate.)

## STEP 3 -- Regime transport reconciliation
- discriminator AUC (train vs val+test) = {disc_auc:.3f}  (0.5 = no shift; higher = stronger covariate shift)
- density-ratio weights w(x): min {w.min():.2f} / median {np.median(w):.2f} / max {w.max():.2f}
- **Method 1 (density ratio, covariate transport)** implied test base rate = **{base_method1:.4f}**
- **Method 2 (recency 3-mo holdout, realized labels)** implied base rate = **{base_method2:.4f}**
- reconciliation gap = **{gap:.4f}**  (train labeled base = 0.1745; val observed = 0.206)

**Interpretation (the methods disagree in DIRECTION, and that is the signal):**
Method 1 says the test book is *safer* (0.166 < 0.1745) -- test applicants are larger / higher-revenue,
which under TRAIN feature relationships maps to lower default. Method 2 says it is *riskier* (0.211).
The held-out **validation** labels (approved dr = {val_real:.4f}) confirm **Method 2**. So the shift is
**concept/label, not pure covariate** (worse macro: observed_revenue_trend_3mo +0.15 -> -0.45): a
density-ratio reweighting of train covariates extrapolates the wrong way. We therefore **anchor the
val/test scoring level to realized recency (Method 2)** -- iso_test is fit on the recency holdout's
true labels -- and treat the **{gap:.4f}** gap as irreducible regime uncertainty that **sizes the
conformal interval width in Prompt 2**. The density-ratio weights are still saved (artifact) for the
writeup's selective-labels / shift discussion.
"""
open("artifacts/backtest_report.md", "w").write(report)

print("\n================ SPINE SUMMARY ================")
print(f"held-out back-fold: log-loss={ll:.5f}  Brier={br:.5f}  meanPred={pd_back.mean():.4f} vs realized={yb.mean():.4f}")
print(f"max |decile calib gap| = {(rel.pred-rel.real).abs().max():.4f}")
print(f"hazard shape reproduced (flat 10-12 + wk13 atom): {shape_ok}")
print(f"validation mean predicted PD (transported) = {val_mean_pd:.4f}  (realized approved {val_real:.4f}; train 0.1745)")
print(f"transport base rates -> method1={base_method1:.4f}  method2={base_method2:.4f}  gap={gap:.4f}")
print(f"artifacts: hazard_model.pkl, applicant_curves.csv ({len(curves)} rows), density_ratio_weights.csv, backtest_report.md")
