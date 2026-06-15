#!/usr/bin/env python3
"""STEP 0 gates: (a) confirm h_9 is a learned hazard (not zeroed); (b) prior_underwriter_score ablation."""
import numpy as np, pandas as pd, pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import importlib.util
spec = importlib.util.spec_from_file_location("bs", "build_spine.py")  # reuse helpers w/o running __main__? build_spine has no guard
# build_spine runs on import; avoid that. Re-declare the few helpers we need instead.

RNG=42; np.random.seed(RNG)
TIME="application_timestamp"; LABEL="default_flag"
CAT=["sector","geography_region","employee_count_bucket","intended_use_of_funds","owner_personal_credit_band","application_channel"]
BANKFEED=["observed_monthly_revenue_avg_3mo","observed_revenue_trend_3mo","observed_revenue_volatility",
          "observed_cash_balance_p10","observed_overdraft_count_3mo","payroll_regularity_score"]
PH1_MAX=9
def wk(d): return np.clip(np.ceil(np.asarray(d,float)/7.0),1,13).astype(int)
def build_features(df, drop_pus=False, onehot=None):
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
    if not drop_pus: num.append("prior_underwriter_score")
    for c in num: X[c]=pd.to_numeric(df[c],errors="coerce")
    d=pd.get_dummies(df[CAT].astype("Int64").astype("category"),prefix=CAT,dummy_na=False)
    X=pd.concat([X,d.astype(int)],axis=1)
    if onehot is not None:
        for c in onehot:
            if c not in X.columns: X[c]=0
        X=X[onehot]
    return X
def hgb(**kw):
    p=dict(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=80,l2_regularization=1.0,
           early_stopping=True,validation_fraction=0.12,random_state=RNG); p.update(kw)
    return HistGradientBoostingClassifier(**p)

tr=pd.read_csv("dataset/train.csv"); tr[TIME]=pd.to_datetime(tr[TIME])
lab=tr[tr[LABEL].notna()].copy().sort_values(TIME).reset_index(drop=True)
lab["y"]=lab[LABEL].astype(int); lab["event_week"]=np.where(lab["y"]==1,wk(lab["days_to_default"]),0)
lab["phase1_default"]=((lab["y"]==1)&(lab["event_week"]<=PH1_MAX)).astype(int)
lab["atom_default"]=((lab["y"]==1)&(lab["event_week"]>=10)).astype(int)
n=len(lab); i1,i2=int(.6*n),int(.8*n)
lab["fold"]=np.where(np.arange(n)<i1,"fit",np.where(np.arange(n)<i2,"cal","back"))
fit,cal=lab[lab.fold=="fit"].copy(),lab[lab.fold=="cal"].copy()

def person_period(sub,Xs):
    ri,wks,ev=[],[],[]
    last=np.where(sub["phase1_default"]==1,sub["event_week"],PH1_MAX).astype(int)
    p1=sub["phase1_default"].values; ew=sub["event_week"].values
    for pos,(lk,p,e) in enumerate(zip(last,p1,ew)):
        for k in range(1,lk+1): ri.append(pos);wks.append(k);ev.append(1 if(p and k==e)else 0)
    PP=Xs.iloc[ri].reset_index(drop=True); PP["age_week"]=wks; return PP,np.array(ev)
def fit_and_loss(drop_pus):
    Xf=build_features(fit,drop_pus); oh=list(Xf.columns)
    PPx,PPy=person_period(fit.reset_index(drop=True),Xf.reset_index(drop=True))
    h1=hgb().fit(PPx,PPy)
    surv=fit["phase1_default"]==0
    atom=hgb(max_iter=300).fit(Xf[surv.values],fit.loc[surv,"atom_default"].values)
    Xc=build_features(cal,drop_pus,onehot=oh); N=len(Xc); S=np.ones(N); F9=np.zeros(N)
    for k in range(1,PH1_MAX+1):
        Xk=Xc.copy(); Xk["age_week"]=k; hk=h1.predict_proba(Xk)[:,1]; F9=F9+S*hk; S=S*(1-hk)
    a13=atom.predict_proba(Xc)[:,1]; F13=F9+S*a13
    return log_loss(cal["y"].values,np.clip(F13,1e-6,1-1e-6))

print("="*70); print("STEP 0a -- h_9 learned? (F8,F9,F10 for 5 sample rows from saved curves)"); print("="*70)
c=pd.read_csv("artifacts/applicant_curves.csv")
samp=c.sample(5,random_state=1)[["applicant_id","F8","F9","F10","F12","F13"]]
for r in samp.itertuples():
    print(f"  {r.applicant_id[:12]}  F8={r.F8:.4f}  F9={r.F9:.4f}  F10={r.F10:.4f}  | F9>F8:{r.F9>r.F8}  F9==F10:{abs(r.F9-r.F10)<1e-12}")
inc9=(c["F9"]-c["F8"]); flat=(c["F10"]-c["F9"]).abs()
print(f"  ACROSS 13,306: week-9 increment F9-F8 mean={inc9.mean():.5f} (min {inc9.min():.5f}, >0 frac {(inc9>0).mean():.3f}); |F10-F9| max={flat.max():.2e}")
print(f"  VERDICT: h_9 {'IS LEARNED (positive wk-9 increment); weeks 10-12 exactly flat -> Prompt1 wording typo, no bug' if (inc9.min()>=0 and inc9.mean()>0 and flat.max()<1e-9) else 'PROBLEM'}")

print("\n"+"="*70); print("STEP 0b -- prior_underwriter_score ablation (cal lifetime-PD log-loss)"); print("="*70)
ll_with=fit_and_loss(drop_pus=False); ll_without=fit_and_loss(drop_pus=True)
delta=ll_without-ll_with
print(f"  WITH  prior_underwriter_score: cal log-loss = {ll_with:.5f}")
print(f"  WITHOUT                      : cal log-loss = {ll_without:.5f}")
print(f"  improvement from keeping it  = {delta:+.5f}")
if delta>0.001: print("  DECISION: KEEP (helps >0.001). NOTE for D Sec.3: it is a prior-model output -> potential SELECTION-BIAS vector.")
else: print("  DECISION: DROP (helps <0.001) -> regenerate curves without it.")
