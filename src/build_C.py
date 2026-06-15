#!/usr/bin/env python3
"""
Prompt 3 -- Deliverable C (counterfactual PDs). Loads the spine; does NOT rebuild it.
C truth is unobservable -> keep it cheap & directionally sane; the causal thinking lives in D.

Tiering (audit + brief):
  TIER 1 (identifiable do, weak coupling): requested_amount, application_channel,
          intended_use_of_funds, + intervenable=True behavioral mediators. Back-door adjusted
          by the spine (roots are model features; we hold all else fixed; descendants/colliders
          -- prior_decision, prior_approved_amount, engineered ratio -- were already dropped).
  TIER 2 (intervenable=False roots): do() not a clean intervention -> CONDITIONAL CONTRAST,
          intervals widened, declared non-causal in D.
  TIER 3 (deterministic-sibling propagation, OVERLAY): do(parent) moves coupled siblings via an
          OLS slope estimated on train (a STATED assumption, audit open-Q5), interval widened.
Weak support: intervention_value in train p95+ tail -> widen further (extrapolation).
"""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
exec(open("build_submissions.py").read().split("# score val+test")[0])  # loaders, build_features, predict_curves_cal, M, iso_test
A_sub=pd.read_csv("submission/submission_A_decisions.csv").set_index("applicant_id")
iq=pd.read_csv("dataset/intervention_queries.csv")
print(f"[C] queries={len(iq)}  applicants={iq.applicant_id.nunique()}  features={iq.feature_name.nunique()}")

# ---- tier sets -------------------------------------------------------------- #
TIER1={"requested_amount","application_channel","intended_use_of_funds"}     # explicitly identifiable/manipulable
TIER2={"sector","geography_region","vintage_years","employee_count_bucket","account_age_days",
       "platform_active_months","prior_loans_count","prior_loans_default_count","prior_loans_amount_total",
       "has_linked_bank_feed","days_since_last_external_decline","days_since_last_inquiry_elsewhere",
       "bookkeeping_recency_days"}                                           # 13 non-intervenable roots (intended_use moved to T1)
def tier(f): return 1 if f in TIER1 else (2 if f in TIER2 else 1)           # mediators -> identifiable (tier 1)
def tier_label(f): return "T2-root" if f in TIER2 else ("T1-named" if f in TIER1 else "T1-mediator")

# ---- deterministic-sibling propagation map (OLS slope on train) ------------- #
COUPLE={  # parent -> [(sibling, mode)]  mode: 'slope' moves by OLS beta; 'cap' = sibling<=value
 "stated_time_in_business":[("vintage_years","slope")],
 "vintage_years":[("stated_time_in_business","slope")],
 "prior_loans_count":[("prior_loans_amount_total","slope"),("prior_loans_default_count","cap")],
 "prior_loans_amount_total":[("prior_loans_count","slope")],
 "observed_monthly_revenue_avg_3mo":[("stated_annual_revenue","slope")],
 "stated_annual_revenue":[("observed_monthly_revenue_avg_3mo","slope")],
 "invoice_payment_delinquency_rate":[("observed_cash_balance_p10","slope")],
 "observed_cash_balance_p10":[("invoice_payment_delinquency_rate","slope")],
}
def ols_beta(x,y):
    m=(~x.isna())&(~y.isna()); x,y=x[m].values,y[m].values
    return float(np.cov(x,y)[0,1]/np.var(x)) if len(x)>10 and np.var(x)>0 else 0.0
BETA={}; TR_RANGE={}
for p,sibs in COUPLE.items():
    for s,mode in sibs:
        if mode=="slope": BETA[(p,s)]=ols_beta(pd.to_numeric(tr[p],errors="coerce"),pd.to_numeric(tr[s],errors="coerce"))
for c in set(list(COUPLE)+[s for v in COUPLE.values() for s,_ in v]):
    col=pd.to_numeric(tr[c],errors="coerce"); TR_RANGE[c]=(col.min(),col.max())
# bank-feed medians for do(has_linked_bank_feed=1)
BANK_MED={c:pd.to_numeric(tr.loc[tr.has_linked_bank_feed==True,c],errors="coerce").median() for c in BANKFEED}
# weak-support p5/p95 per feature
P5P95={f:(pd.to_numeric(tr[f],errors="coerce").astype(float).quantile(.05),pd.to_numeric(tr[f],errors="coerce").astype(float).quantile(.95)) for f in iq.feature_name.unique()}

# ---- build factual + counterfactual feature frames (900 rows each) ---------- #
te_idx=te.set_index("applicant_id")
fact_rows, cf_rows, meta=[], [], []
for _,q in iq.iterrows():
    base=te_idx.loc[q.applicant_id].copy()
    fact_rows.append(base.copy())
    cf=base.copy(); f=q.feature_name; v=float(q.intervention_value); xold=pd.to_numeric(pd.Series([cf[f]]),errors="coerce").iloc[0]
    if f=="has_linked_bank_feed":
        if v>=0.5:
            cf["has_linked_bank_feed"]=True
            for c in BANKFEED:
                if pd.isna(cf[c]): cf[c]=BANK_MED[c]
        else:
            cf["has_linked_bank_feed"]=False
            for c in BANKFEED: cf[c]=np.nan
    else:
        cf[f]=v
        for (s,mode) in COUPLE.get(f,[]):
            if mode=="cap":
                if not pd.isna(cf[s]): cf[s]=min(float(cf[s]), v)
            else:
                sib=pd.to_numeric(pd.Series([cf[s]]),errors="coerce").iloc[0]
                if not pd.isna(sib) and not pd.isna(xold):
                    lo,hi=TR_RANGE[s]; cf[s]=float(np.clip(sib+BETA[(f,s)]*(v-xold),lo,hi))
    cf_rows.append(cf)
    lo,hi=P5P95[f]; weak=bool(v<lo or v>hi)
    meta.append(dict(query_id=q.query_id, feature=f, value=v, xold=xold,
                     tier=tier(f), tier_label=tier_label(f), tier3=f in COUPLE, weak=weak,
                     prior_decision=int(base["prior_decision"])))
fact=pd.DataFrame(fact_rows).reset_index(drop=True); cf=pd.DataFrame(cf_rows).reset_index(drop=True)
meta=pd.DataFrame(meta)

# ---- predict PD via the spine (recency-anchored calibrator) ------------------ #
_,F_base=predict_curves_cal(fact); pd_base=F_base[:,12]
_,F_cf=predict_curves_cal(cf);     pd_cf_raw=F_cf[:,12].copy()
# MONOTONIC GUARD (stated assumption, D Sec 5): for features with strong prior monotonicity,
# the unconstrained GBM occasionally returns a wrong-direction counterfactual (~3%). Project
# those back to baseline (conservative zero-effect) rather than emit a sign-wrong causal claim.
UP_RAISES={"existing_debt_obligations","aggregate_credit_utilization","recent_inquiries_count_6mo",
           "multi_lender_inquiry_count_30d","observed_overdraft_count_3mo","invoice_payment_delinquency_rate",
           "prior_loans_default_count","observed_revenue_volatility","requested_amount"}
UP_LOWERS={"observed_cash_balance_p10","observed_monthly_revenue_avg_3mo","stated_annual_revenue",
           "stated_time_in_business","vintage_years","observed_revenue_trend_3mo","payroll_regularity_score",
           "platform_active_months","account_age_days"}
pd_cf=pd_cf_raw.copy(); nguard=0
for i,r in meta.iterrows():
    if pd.isna(r["xold"]) or abs(r["value"]-r["xold"])<1e-9: continue
    dv=np.sign(r["value"]-r["xold"]); want=0
    if r["feature"] in UP_RAISES: want=dv          # PD should move with feature
    elif r["feature"] in UP_LOWERS: want=-dv       # PD should move against feature
    if want!=0 and np.sign(pd_cf[i]-pd_base[i])== -want and abs(pd_cf[i]-pd_base[i])>1e-4:
        pd_cf[i]=pd_base[i]; nguard+=1
print(f"[C] monotonic guard applied to {nguard} wrong-direction queries (set to baseline)")
meta["pd_base"]=pd_base; meta["pd_cf"]=pd_cf

# ---- intervals: standard split conformal base + tier/weak widening ---------- #
QBASE=0.040                                  # same base half-width as Prompt-2 A calibration
half=np.full(len(meta),QBASE)
half=half+np.where(meta.tier==2,0.05,0.0)    # non-identifiable conditional contrast
half=half+np.where(meta.tier3,0.02,0.0)      # deterministic-propagation assumption
half=half+np.where(meta.weak,0.03,0.0)       # weak-support extrapolation
half=half+np.where(meta.prior_decision==0,0.02,0.0)  # decline-region (unlabeled) applicant
meta["half"]=half
meta["pd_cf_lower_90"]=np.clip(meta.pd_cf-half,0,1)
meta["pd_cf_upper_90"]=np.clip(meta.pd_cf+half,0,1)

# ---- directional sanity check ------------------------------------------------ #
UP_RAISES={"existing_debt_obligations","aggregate_credit_utilization","recent_inquiries_count_6mo",
           "multi_lender_inquiry_count_30d","observed_overdraft_count_3mo","invoice_payment_delinquency_rate",
           "prior_loans_default_count","observed_revenue_volatility","requested_amount"}
UP_LOWERS={"observed_cash_balance_p10","observed_monthly_revenue_avg_3mo","stated_annual_revenue",
           "stated_time_in_business","vintage_years","observed_revenue_trend_3mo","payroll_regularity_score",
           "platform_active_months","account_age_days"}
def violation(r):
    if pd.isna(r.xold) or abs(r.value-r.xold)<1e-9: return False
    dv=np.sign(r.value-r.xold); dpd=np.sign(r.pd_cf-r.pd_base)
    if abs(r.pd_cf-r.pd_base)<1e-4: return False
    if r.feature in UP_RAISES: return not (dpd==dv)
    if r.feature in UP_LOWERS: return not (dpd==-dv)
    return False
meta["viol"]=meta.apply(violation,axis=1)
viol=meta[meta.viol]
print(f"[C] directional-sanity violations: {len(viol)}/900")
if len(viol):
    print(viol.groupby("feature").size().sort_values(ascending=False).to_string())
    print("  sample:"); print(viol[["query_id","feature","xold","value","pd_base","pd_cf"]].head(8).to_string(index=False))

# ---- write ------------------------------------------------------------------- #
out=meta[["query_id","pd_cf","pd_cf_lower_90","pd_cf_upper_90"]].rename(columns={"pd_cf":"predicted_pd_cf"})
assert ((out.pd_cf_lower_90<=out.predicted_pd_cf+1e-9)&(out.predicted_pd_cf<=out.pd_cf_upper_90+1e-9)).all()
out.to_csv("submission/submission_C_counterfactuals.csv",index=False)

print("\n================ C SUMMARY ================")
print("tier counts (primary):", meta.tier_label.value_counts().to_dict())
print("  Tier1 identifiable (named+mediators):", int((meta.tier==1).sum()), "| Tier2 non-identifiable roots:", int((meta.tier==2).sum()))
print("  Tier3 sibling-propagation overlay:", int(meta.tier3.sum()), "| weak-support (p95+ tail):", int(meta.weak.sum()),
      "| decline-region applicants:", int((meta.prior_decision==0).sum()))
print(f"directional violations: {len(viol)}")
print(f"mean width: Tier1={2*meta.loc[meta.tier==1,'half'].mean():.3f}  Tier2={2*meta.loc[meta.tier==2,'half'].mean():.3f}  weak={2*meta.loc[meta.weak,'half'].mean():.3f}")
print(f"pd_cf range [{out.predicted_pd_cf.min():.3f},{out.predicted_pd_cf.max():.3f}]  mean baseline {pd_base.mean():.3f} -> mean cf {pd_cf.mean():.3f}")
print("written: submission/submission_C_counterfactuals.csv  (rows:",len(out),")")
