#!/usr/bin/env python3
"""
Prompt 2 -- Deliverables A & B from the saved spine. Writes submission/ folder.
Loads artifacts/hazard_model.pkl (regenerated WITHOUT prior_underwriter_score per STEP 0)
and artifacts/applicant_curves.csv. Does NOT rebuild the hazard spine.

Scoring economics (hackathon-brief.pdf p.8-9), not re-derived:
  repaid:        NPV = F + R*r*T/365
  default day t*: NPV = F + D*draws + rec - R
  R=requested_amount, r=0.35, T=60, F=0.03R, D=R*(1+r*T/365)/T.
  draws: NPV_MODE="capped" -> min(t*-1, T-1) <= 59 (DEFAULT; organizer clarified days 60-90 are
    a late-payment window where the owed balance is recovered, NOT extra ACH draws -- the literal
    D*(t*-1) credits 89 draws at the day-90 atom, making a defaulter beat a clean repayment).
    NPV_MODE="literal" reproduces the brief formula verbatim (held as a fallback set).
  recovery: modeled per loan for EARLY defaults (~12% of R); the day-90 ATOM recovers ~0 in the
    REAL labels (residual balance, not collateral), so the atom branch uses the empirical atom
    recovery rate, NOT the early-default model average (PHASE-0 finding: model over-credits atoms).
  approve iff E[NPV]>0 (NPV sign, not a flat PD threshold).
  B: CDR_{w,a} = mean over the TEST-ONLY approved set A_w of F_i(7a) (organizer ruling: B test-only).
  Intervals: STANDARD split conformal (NOT Tibshirani-2019 weighted; covariate-shift assumption
    violated, residual gap 0.0037). We optimise an interval-score-LIKE tradeoff but attribute no
    named metric to the scorer.
"""
import os, pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
warnings = __import__("warnings"); warnings.filterwarnings("ignore")
RNG = 42; np.random.seed(RNG)
NPV_MODE = os.environ.get("NPV_MODE", "capped")     # {"capped","literal"} -- draws_collected switch
OUTDIR   = os.environ.get("OUTDIR", "submission")    # default vs submission_literal
assert NPV_MODE in ("capped", "literal"), NPV_MODE
os.makedirs(OUTDIR, exist_ok=True)
print(f"[cfg] NPV_MODE={NPV_MODE}  OUTDIR={OUTDIR}")

TIME="application_timestamp"; LABEL="default_flag"; PH1_MAX=9
CAT=["sector","geography_region","employee_count_bucket","intended_use_of_funds","owner_personal_credit_band","application_channel"]
BANKFEED=["observed_monthly_revenue_avg_3mo","observed_revenue_trend_3mo","observed_revenue_volatility",
          "observed_cash_balance_p10","observed_overdraft_count_3mo","payroll_regularity_score"]
R_APR, T_TERM, FEE = 0.35, 60, 0.03
def wk(d): return np.clip(np.ceil(np.asarray(d,float)/7.0),1,13).astype(int)

# ---- feature builder: MUST match the regenerated spine (no prior_underwriter_score) ----
def build_features(df, onehot=None):
    X=pd.DataFrame(index=df.index)
    X["miss_ext_decline"]=df["days_since_last_external_decline"].isna().astype(int)
    X["miss_inq_elsewhere"]=df["days_since_last_inquiry_elsewhere"].isna().astype(int)
    X["no_bank_feed"]=(~df["has_linked_bank_feed"].astype(bool)).astype(int)
    X["has_linked_bank_feed"]=df["has_linked_bank_feed"].astype(int)
    num=["vintage_years","stated_annual_revenue","stated_time_in_business","requested_amount",*BANKFEED,
         "aggregate_credit_utilization","recent_inquiries_count_6mo","existing_debt_obligations",
         "days_since_last_external_decline","account_age_days","platform_active_months","bookkeeping_recency_days",
         "invoice_payment_delinquency_rate","prior_loans_count","prior_loans_default_count","prior_loans_amount_total",
         "multi_lender_inquiry_count_30d","days_since_last_inquiry_elsewhere","repeat_application_count"]
    for c in num: X[c]=pd.to_numeric(df[c],errors="coerce")
    d=pd.get_dummies(df[CAT].astype("Int64").astype("category"),prefix=CAT,dummy_na=False)
    X=pd.concat([X,d.astype(int)],axis=1)
    if onehot is not None:
        for c in onehot:
            if c not in X.columns: X[c]=0
        X=X[onehot]
    return X

art=pickle.load(open("artifacts/hazard_model.pkl","rb"))
M=art["model"]; iso_test=art["iso_test"]; GAP=art["transport"]["gap"]
print(f"[load] use_opaque={art['use_opaque']}  transport_gap(corrected)={GAP:.4f}  (Prompt-1 stale gap was 0.045)")

def predict_curves_cal(df):
    """Calibrated cumulative incidence Fcal[:,1..13] and raw hazard H, matching saved curves."""
    X=build_features(df, onehot=M["onehot"]); N=len(X)
    H=np.zeros((N,14))
    for k in range(1,PH1_MAX+1):
        Xk=X.copy(); Xk["age_week"]=k; H[:,k]=M["h1"].predict_proba(Xk)[:,1]
    a13=M["atom"].predict_proba(X)[:,1]
    S=np.ones(N); F=np.zeros((N,14))
    for k in range(1,PH1_MAX+1): F[:,k]=F[:,k-1]+S*H[:,k]; S=S*(1-H[:,k])
    for k in (10,11,12): F[:,k]=F[:,k-1]
    H[:,13]=a13; F[:,13]=F[:,12]+S*a13
    Fr=F[:,1:14]
    Fc=np.clip(iso_test.transform(Fr.reshape(-1)).reshape(Fr.shape),0,1)
    Fc=np.maximum.accumulate(Fc,axis=1)
    return H[:,1:14], Fc

# ----------------------------------------------------------------------------- #
tr=pd.read_csv("dataset/train.csv"); va=pd.read_csv("dataset/validation.csv"); te=pd.read_csv("dataset/test.csv")
for d in (tr,va,te): d[TIME]=pd.to_datetime(d[TIME])
lab=tr[tr[LABEL].notna()].copy().sort_values(TIME).reset_index(drop=True)
lab["y"]=lab[LABEL].astype(int); lab["event_week"]=np.where(lab["y"]==1,wk(lab["days_to_default"]),0)
n=len(lab); i1,i2=int(.6*n),int(.8*n)
lab["fold"]=np.where(np.arange(n)<i1,"fit",np.where(np.arange(n)<i2,"cal","back"))
cal_f=lab[lab.fold=="cal"].copy(); back=lab[lab.fold=="back"].copy()

# ----------------------------------------------------------------------------- #
# A.1 -- recovery model rec_i (observable features only; target=final_recovered_amount)
# ----------------------------------------------------------------------------- #
rec_feats=["requested_amount","existing_debt_obligations","aggregate_credit_utilization",
           "observed_cash_balance_p10","prior_loans_default_count","owner_personal_credit_band","sector"]
dfl=tr[(tr[LABEL]==1)&(tr["final_recovered_amount"].notna())].copy()        # train defaulters
Xr=dfl[rec_feats].apply(pd.to_numeric,errors="coerce")
rec_model=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,
            min_samples_leaf=60,random_state=RNG).fit(Xr, dfl["final_recovered_amount"].values)
def predict_rec(df):
    r=rec_model.predict(df[rec_feats].apply(pd.to_numeric,errors="coerce"))
    return np.clip(r,0,df["requested_amount"].values)                        # 0 <= rec <= R
print(f"[recovery] train defaulters={len(dfl)}  realized rec mean={dfl['final_recovered_amount'].mean():.0f} "
      f"({(dfl['final_recovered_amount']/dfl['requested_amount']).mean():.3f} of R)")
# PHASE-1b: the day-90 ATOM recovers ~0 in the real labels (residual balance, not collateral);
# the blind recovery model over-credits it. Size atom recovery to the empirical atom rate instead.
_atom=tr[(tr[LABEL]==1)&(tr["days_to_default"]==90)&(tr["final_recovered_amount"].notna())]
ATOM_REC_RATE=float((_atom["final_recovered_amount"]/_atom["requested_amount"]).mean())
print(f"[recovery] ATOM (t*=90) empirical recovery rate={ATOM_REC_RATE:.4f} of R (n={len(_atom)}) "
      f"vs early-model average; atom NPV branch uses rate*R, early branches keep the model")

# ----------------------------------------------------------------------------- #
# A.2 -- timing-aware E[NPV] from per-week default MASS
# ----------------------------------------------------------------------------- #
REP_DAY={k:7*k-3 for k in range(1,10)}; REP_DAY[13]=90                       # representative default day
print("[NPV] representative default day per week:", {**{k:REP_DAY[k] for k in range(1,10)}, 13:90},
      "| weeks 10-12 carry zero mass")
def draws_collected(rep_day):
    # capped (DEFAULT, economically correct): no ACH draws after the T=60 term -> cap at T-1=59.
    # literal (fallback): brief formula D*(t*-1) with NO cap (=> 89 at the day-90 atom).
    return (rep_day-1) if NPV_MODE=="literal" else min(rep_day-1, T_TERM-1)
def enpv(df, Fc, rec):
    R=df["requested_amount"].values.astype(float)
    D=R*(1+R_APR*T_TERM/365)/T_TERM
    npv_repaid=FEE*R + R*R_APR*T_TERM/365
    S13=1.0-Fc[:,12]
    E=S13*npv_repaid
    mass=np.diff(np.concatenate([np.zeros((len(R),1)),Fc],axis=1),axis=1)    # mass[:,k-1]=F_k-F_{k-1}
    rep_days={**REP_DAY}
    for k in list(range(1,10))+[13]:
        mk=mass[:,k-1]
        rec_k = ATOM_REC_RATE*R if k==13 else rec   # atom recovers ~0 (label-grounded); early uses model
        npv_def=FEE*R + D*draws_collected(rep_days[k]) + rec_k - R
        E=E+mk*npv_def
    return E, npv_repaid

# score val+test
val_curves=pd.read_csv("artifacts/applicant_curves.csv")                     # has F1..F13 (calibrated) for val+test
Fcols=[f"F{k}" for k in range(1,14)]
def assemble(df, split):
    sub=val_curves[val_curves.split==split].set_index("applicant_id").loc[df["applicant_id"].values]
    Fc=sub[Fcols].values
    rec=predict_rec(df)
    E,npv_rep=enpv(df,Fc,rec)
    out=pd.DataFrame({"applicant_id":df["applicant_id"].values,
                      "predicted_pd":Fc[:,12], "ENPV":E,
                      "decision":(E>0).astype(int),
                      "prior_decision":df["prior_decision"].values,
                      "no_bank_feed":(~df["has_linked_bank_feed"].astype(bool)).astype(int).values,
                      "cohort_ts":df[TIME].values})
    return out, Fc
A_val,Fc_val=assemble(va,"validation"); A_te,Fc_te=assemble(te,"test")
A=pd.concat([A_val,A_te],ignore_index=True)
print(f"[A] approve rate={A.decision.mean():.3f}  avg predicted_pd={A.predicted_pd.mean():.4f} "
      f"(val labeled base 0.206; train 0.1745)  mean E[NPV] on approved={A.loc[A.decision==1,'ENPV'].mean():.1f}")

# ----------------------------------------------------------------------------- #
# CALIBRATION -- standard split conformal, grouped coverage; width tuned on VAL (target regime)
# ----------------------------------------------------------------------------- #
def grouped_cov(phat,y,half,K=20):
    b=pd.qcut(phat,K,labels=False,duplicates="drop")
    g=pd.DataFrame({"b":b,"p":phat,"y":y,"h":half}).groupby("b").agg(p=("p","mean"),y=("y","mean"),h=("h","mean"))
    return float(((g.y>=g.p-g.h)&(g.y<=g.p+g.h)).mean())
# predicted PD on held-out folds (iso_test = submission calibrator)
_,Fc_back=predict_curves_cal(back); p_back=Fc_back[:,12]; y_back=back["y"].values
_,Fc_vl=predict_curves_cal(va[va[LABEL].notna()]); p_vl=Fc_vl[:,12]; y_vl=va.loc[va[LABEL].notna(),LABEL].astype(int).values
grid=np.round(np.arange(0.0,0.30,0.002),3)
# Validation is the DESIGNATED calibration set (dataset README) AND the test regime, so we
# tune the split-conformal width on it directly -- this SUPERSEDES a separate train->test
# drift margin (adding the stale 0.045 / corrected 0.0037 gap on top would double-count).
# K=10 coverage is coarse (10% steps) and val alone at K=20 is too noisy (127/bin). We target
# ~0.89 JOINTLY on val at decile resolution AND on pooled back+val at K=20 (~645/bin, the
# resolution matching the 13,306 scored set). Minimal width meeting both >=0.88.
p_pool=np.concatenate([p_back,p_vl]); y_pool=np.concatenate([y_back,y_vl])
def _ok(q): return (grouped_cov(p_vl,y_vl,np.full(len(p_vl),q),K=10)>=0.88
                    and grouped_cov(p_pool,y_pool,np.full(len(p_pool),q),K=20)>=0.88)
q_base=next((q for q in grid if _ok(q)),0.30)
q_back=next((q for q in grid if grouped_cov(p_back,y_back,np.full(len(p_back),q),K=10)>=0.88),0.30)
print(f"[A-cal] q_base={q_base:.3f} (min width s.t. val_K10>=.88 AND pooled_K20>=.88) | back-fold needs only "
      f"q_back={q_back:.3f} (train regime) | no stale-0.045 margin (val-tuned in-regime; corrected gap {GAP:.4f})")
# per-applicant half-width: base (approved-like) + decline widen (unlabeled extrapolation) + low-density widen
DECL_EXTRA=0.05; LOWDENS_EXTRA=0.02
def half_width_A(df_flag):
    h=np.full(len(df_flag),q_base,float)
    h=h+np.where(df_flag["prior_decision"].values==0, DECL_EXTRA, 0.0)        # decline region wider
    h=h+np.where(df_flag["no_bank_feed"].values==1, LOWDENS_EXTRA, 0.0)       # low-info wider
    return h
A["half"]=half_width_A(A)
A["pd_lower_90"]=np.clip(A.predicted_pd-A.half,0,1)
A["pd_upper_90"]=np.clip(A.predicted_pd+A.half,0,1)
# coverage on held-out using EXACT submission half-widths (approved region => base width)
covA_back=grouped_cov(p_back,y_back,np.full(len(p_back),q_base),K=10)
covA_val =grouped_cov(p_vl ,y_vl ,np.full(len(p_vl ),q_base),K=10)
covA_pool=grouped_cov(p_pool,y_pool,np.full(len(p_pool),q_base),K=20)
wd_appr=A.loc[A.prior_decision==1,"half"].mean()*2; wd_decl=A.loc[A.prior_decision==0,"half"].mean()*2
pd_appr_region=A.loc[A.prior_decision==1,"predicted_pd"].mean(); pd_decl_region=A.loc[A.prior_decision==0,"predicted_pd"].mean()
print(f"[A-cal] grouped coverage: val_K10={covA_val:.3f} pooled_K20={covA_pool:.3f} back_K10={covA_back:.3f} | mean width approved={wd_appr:.3f} decline={wd_decl:.3f}")
print(f"[A] predicted_pd by region: prior-APPROVED={pd_appr_region:.4f} (~val 0.206) | prior-DECLINED={pd_decl_region:.4f} (unlabeled, riskier)")

# ----------------------------------------------------------------------------- #
# DELIVERABLE B -- approved-cohort cumulative default trajectory + intervals
# ----------------------------------------------------------------------------- #
cw=pd.read_csv("dataset/cohort_week_definitions.csv",parse_dates=["start_date","end_date"])
def cohort_of(ts):
    ts=pd.to_datetime(ts); c=np.full(len(ts),-1)
    for _,r in cw.iterrows(): c[(ts>=r.start_date)&(ts<=r.end_date)]=int(r.cohort_week)
    return c
allF=np.vstack([Fc_val,Fc_te])
A["cohort_week"]=cohort_of(A["cohort_ts"])
pop_pd=A["predicted_pd"].mean()

# B interval method tuned on pseudo-cohorts of the back fold (trajectory coverage)
np.random.seed(RNG); G=13
pc=np.random.randint(0,G,size=len(back))
cells=[]
for g in range(G):
    idx=np.where(pc==g)[0]
    if len(idx)<30: continue
    Fc_g=Fc_back[idx]; ew_g=back["event_week"].values[idx]; y_g=back["y"].values[idx]
    for a in range(1,14):
        pred=Fc_g[:,a-1].mean(); real=np.mean((y_g==1)&(ew_g<=a)); cells.append((len(idx),pred,real))
cells=np.array(cells,float)
def covB(kB,aB):
    n_,pred,real=cells[:,0],cells[:,1],cells[:,2]
    h=kB*np.sqrt(np.clip(pred*(1-pred),1e-9,None)/n_)+aB
    return float(np.mean((real>=pred-h)&(real<=pred+h))), float(np.mean(2*h))
best=None
for kB in [1.0,1.28,1.645,2.0,2.5,3.0]:
    for aB in [0.0,0.005,0.01,0.02,0.03]:
        cov,wid=covB(kB,aB)
        if cov>=0.90 and (best is None or wid<best[2]): best=(kB,aB,wid,cov)
kB,aB,_,covB_back=best
print(f"[B-cal] pseudo-cohort coverage tuned: kB={kB} aB={aB} -> back coverage={covB_back:.3f}")

# build the 169-row submission B from template
tmpl=pd.read_csv("dataset/submission_B_template.csv")
# ORGANIZER RULING: Deliverable B is TEST-ONLY. A_w = applicants YOU approved drawn from test.csv
# only (validation is for calibration, not part of B). A/allF are in concat(val, test) row order.
TEST_IDS=set(te["applicant_id"]); test_mask_all=A["applicant_id"].isin(TEST_IDS).values
appr=A[(A.decision==1) & (A["applicant_id"].isin(TEST_IDS))].copy(); appr_idx=appr.index.values
allF_appr=allF[appr_idx]
print(f"[B] TEST-ONLY approved set: {len(appr)} (was {(A.decision==1).sum()} val+test)")
rowsB=[]
cohort_levels={}
for _,row in tmpl.iterrows():
    w=int(row.cohort_week); a=int(row.loan_age_weeks)
    mask=appr["cohort_week"].values==w
    if mask.sum()==0:                                  # fallback: full TEST cohort (rare/empty approved)
        mask=(A["cohort_week"].values==w)&test_mask_all; Fsrc=allF
    else: Fsrc=allF_appr
    cdr=float(Fsrc[mask,a-1].mean()); nA=int(mask.sum())
    h=kB*np.sqrt(max(cdr*(1-cdr),1e-9)/max(nA,1))+aB
    rowsB.append((w,a,cdr,max(0.0,cdr-h),min(1.0,cdr+h)))
    if a==13: cohort_levels[w]=(cdr,nA)
B=pd.DataFrame(rowsB,columns=["cohort_week","loan_age_weeks","cumulative_default_rate","cdr_lower_90","cdr_upper_90"])
# enforce per-cohort monotonicity of the point curve AND bounds, keep lower<=point<=upper
B=B.sort_values(["cohort_week","loan_age_weeks"]).reset_index(drop=True)
for w in range(1,14):
    m=B.cohort_week==w
    B.loc[m,"cumulative_default_rate"]=np.maximum.accumulate(B.loc[m,"cumulative_default_rate"].values)
B["cdr_lower_90"]=np.minimum(B["cdr_lower_90"],B["cumulative_default_rate"])
B["cdr_upper_90"]=np.maximum(B["cdr_upper_90"],B["cumulative_default_rate"])
B[["cdr_lower_90","cumulative_default_rate","cdr_upper_90"]]=B[["cdr_lower_90","cumulative_default_rate","cdr_upper_90"]].clip(0,1)

appr_cdr13=np.mean([cohort_levels[w][0] for w in cohort_levels])
print(f"[B] approved-cohort mean lifetime CDR={appr_cdr13:.4f}  vs population predicted PD={pop_pd:.4f} "
      f"(approved should be LOWER)")

# ----------------------------------------------------------------------------- #
# WRITE
# ----------------------------------------------------------------------------- #
A_out=A[["applicant_id","decision","predicted_pd","pd_lower_90","pd_upper_90"]].copy()
A_out.to_csv(f"{OUTDIR}/submission_A_decisions.csv",index=False)
B[["cohort_week","loan_age_weeks","cumulative_default_rate","cdr_lower_90","cdr_upper_90"]].to_csv(
    f"{OUTDIR}/submission_B_trajectory.csv",index=False)

print("\n================ SUBMISSION SUMMARY (A+B) ================")
print(f"A rows={len(A_out)} (expect 13306)  approve_rate={A.decision.mean():.3f}")
print(f"avg predicted_pd={A.predicted_pd.mean():.4f}  (target ~0.206 regime, NOT 0.1745)")
print(f"mean E[NPV] on APPROVED set={A.loc[A.decision==1,'ENPV'].mean():.1f}  (>0 required)  "
      f"| mean E[NPV] on DECLINED={A.loc[A.decision==0,'ENPV'].mean():.1f}")
print(f"B rows={len(B)} (expect 169)  approved-cohort lifetime CDR={appr_cdr13:.4f} < population {pop_pd:.4f}: {appr_cdr13<pop_pd}")
print(f"A interval mean width: approved={wd_appr:.3f}  decline={wd_decl:.3f}  (decline wider: {wd_decl>wd_appr})")
print(f"A held-out grouped coverage: val_K10={covA_val:.3f}  pooled_K20={covA_pool:.3f}  back_K10={covA_back:.3f}  (target 0.87-0.93)")
print(f"B pseudo-cohort coverage (back)={covB_back:.3f}  mean B width={(B.cdr_upper_90-B.cdr_lower_90).mean():.4f}")
print(f"written: {OUTDIR}/submission_A_decisions.csv, {OUTDIR}/submission_B_trajectory.csv")
