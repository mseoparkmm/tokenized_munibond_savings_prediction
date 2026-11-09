"""Data preparation and descriptive diagnostics, shared by the reproducible pipeline.
Frozen conceptual gate from the previous research design.
"""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f as fdist
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'input_source.xlsx'
COMP={'Underwriter':'underwriter','Bond Counsel':'bond counsel','Financial Advisor':'financial advisor','Paying Agent':'paying agent','Rating Agency Fee':'rating agent','Other Costs':'other'}
VARS=['Principal Amount','Credit Rating','Sale Type','Government Type','Issue Purpose','Fiscal Year','Fed Rate','Rating Agency','Closing Month']
# Ex ante service-fee mechanism screen. This is conceptual, not statistical pruning.
GATE={
 'Underwriter':{'Principal Amount':'Scale of underwriting effort and spread','Credit Rating':'Credit risk and distribution','Sale Type':'Competitive versus negotiated distribution structure','Government Type':'Issuer organization and procurement proxy','Issue Purpose':'Debt structure and placement complexity','Fiscal Year':'Nominal fee drift and market structure','Fed Rate':'Debt market conditions'},
 'Bond Counsel':{'Principal Amount':'Complexity and transaction scale','Sale Type':'Legal process differs by sale method','Government Type':'Legal and governance structure','Issue Purpose':'Legal requirements of specific purposes','Fiscal Year':'Nominal professional fee drift'},
 'Financial Advisor':{'Principal Amount':'Transaction scale','Credit Rating':'Financing difficulty proxy; indirect','Sale Type':'Advisory process differs by sale method','Government Type':'Issuer advisory/procurement practices','Issue Purpose':'Structuring complexity','Fiscal Year':'Nominal professional fee drift','Fed Rate':'Market complexity proxy; indirect'},
 'Paying Agent':{'Principal Amount':'Potential issue-volume fee schedules; weak','Fiscal Year':'Possible changes to fixed registrar fees'},
 'Rating Agency Fee':{'Principal Amount':'Rating fee schedules linked to issuance size','Credit Rating':'Analytical complexity/risk proxy','Government Type':'Issuer and credit structure','Issue Purpose':'Rating analytical complexity','Fiscal Year':'Fee schedule drift','Rating Agency':'Identity of fee-charging provider'},
 'Other Costs':{'Principal Amount':'Transaction scale and residual professional work','Sale Type':'Residual issuance activities can vary by sale structure','Government Type':'Administrative/procurement requirements','Issue Purpose':'Residual deal complexity','Fiscal Year':'Nominal fee drift'}
}
EXCL={
 'Rating Agency':'Fee is charged by rating agency, but no direct underwriter/advisor/counsel/registrar fee mechanism is supported',
 'Closing Month':'Calendar month alone does not establish a fee mechanism; agreement-to-close month can be unknown ex ante',
 'Fed Rate':'No sufficiently direct link to this service fee; year can represent nominal secular changes',
 'Credit Rating':'Rating level does not directly price this professional/administrative service',
 'Sale Type':'Sale route does not directly determine this registrar/rating service fee',
 'Issue Purpose':'No clear direct fee pathway for this specific service',
 'Government Type':'Issuer legal category is not a defensible direct predictor for this service without evidence'
}
NUM=['Principal Amount','Credit Rating','Fiscal Year','Fed Rate']
COL={'Principal Amount':'logP','Credit Rating':'RatingScore','Sale Type':'SaleType','Government Type':'GovernmentType','Issue Purpose':'PurposeGroup','Fiscal Year':'FiscalYearIssuance','Fed Rate':'Fed Rate','Rating Agency':'RatingAgency','Closing Month':'ClosingMonth'}
CATEG=['Sale Type','Government Type','Issue Purpose','Rating Agency','Closing Month']
RATING={'AAA':1,'Aaa':1,'AA+':2,'Aa1':2,'AA':3,'Aa2':3,'AA-':4,'Aa3':4,'A+':5,'A1':5,'A':6,'A2':6,'A-':7,'A3':7,'BBB+':8,'Baa1':8,'BBB':9,'Baa2':9,'BBB-':10,'Baa3':10}
MAIN_PURPOSE={'GeneralPurpose','Transportation','WaterRelated','Recreation','PublicSafety'}

def prepare():
 d0=pd.read_excel(DATA).copy()
 fee_cols=list(COMP.values())
 assert len(d0)==479
 assert d0[fee_cols].notna().all().all()
 assert (d0[fee_cols]>=0).all().all()
 # Exclusive screening, deliberately fixed before any new outcome inspection.
 is_private=d0.SaleType.eq('PrivatePlacement')
 is_nonpositive=d0[fee_cols].le(0).any(axis=1)
 is_rating_missing=d0['Credit Rating'].isna()
 stage=np.select([is_private,is_nonpositive,is_rating_missing],
                ['PrivatePlacement','NonpositiveAnyOfSix','MissingCreditRating'],default='Retained')
 record=d0[['IssuerName','IssuanceName','FiscalYearIssuance','ActualPar','SaleType','Credit Rating',*fee_cols]].copy()
 record.insert(0,'SourceExcelRow',np.arange(len(d0))+2)
 record['ExclusionStage']=stage
 record['AnyZeroFee']=d0[fee_cols].eq(0).any(axis=1)
 record.to_csv(ROOT/'00_data_filter_479.csv',index=False)
 counts=record.ExclusionStage.value_counts().to_dict()
 assert counts=={'Retained':436,'NonpositiveAnyOfSix':31,'PrivatePlacement':9,'MissingCreditRating':3}, counts
 mask=(stage=='Retained')
 d=d0.loc[mask].copy()
 d['SourceExcelRow']=np.flatnonzero(mask)+2
 # IDs from former 465-model generated BEFORE exclusion of zero-fee issues.
 old_mask=(~is_private & ~is_rating_missing)
 old_issue_id=np.cumsum(old_mask.to_numpy())
 d['Original465IssueId']=old_issue_id[mask]
 d.reset_index(drop=True,inplace=True)
 d['logP']=np.log(d['ActualPar']/1e6)
 d['RatingScore']=d['Credit Rating'].map(RATING)
 if d['RatingScore'].isna().any():raise ValueError('Unmapped or missing rating')
 d['RatingAgency']=d['Credit rating Agency '].fillna('Unknown').astype(str)
 d['PurposeGroup']=d['IssuePurpose'].map(lambda x:x if x in MAIN_PURPOSE else 'OtherPurpose')
 d['ClosingMonth']=pd.to_datetime(d['ClosingDate'],errors='coerce').dt.month
 d['IssueId']=np.arange(1,len(d)+1)
 if len(d)!=436 or int((d.FiscalYearIssuance<=2021).sum())!=249:raise ValueError('Unexpected sample')
 if not (d[fee_cols].gt(0).all().all() and d['Credit Rating'].notna().all()):raise ValueError('Filtering failure')
 if not np.allclose(d[fee_cols].sum(axis=1),d['Total Cost'],atol=.01):raise ValueError('Total mismatch')
 aud={'raw_n':479,'private_removed':9,'any_zero_nonprivate_removed':31,'remaining_all_six_positive':439,
      'additional_missing_credit_rating_removed':3,'model_n':436,'dev_before_fy2022':249,
      'outer_fy2022_2025_n':187,'test_year_counts':{str(y):int(sum(d.FiscalYearIssuance==y)) for y in range(2022,2026)},
      'zero_handling':'Exclude any issuance with one or more of six component fees <=0 from ALL six model fits and all validation folds.',
      'exclusion_order':['PrivatePlacement','NonpositiveAnyOfSix','MissingCreditRating'],
      'fee_components':fee_cols,
      'note':'The earlier Step1 439-fee-positive sample contained 3 missing credit ratings; this complete-case model uses 436.'}
 (ROOT/'00_sample_audit.json').write_text(json.dumps(aud,indent=2))
 d.to_csv(ROOT/'analysis_data_436.csv',index=False)
 return d

def orthogonal_X(df, vars, with_p=True, with_year=False, interactions=False):
 # Regression diagnostics only; use entire df for in-development associations.
 X=pd.DataFrame(index=df.index)
 if with_p:
  z=df.logP.to_numpy()-df.logP.mean()
  X['logP']=z;X['logP2']=z*z;X['logP3']=z*z*z
 if with_year and 'Fiscal Year' not in vars:
  X['YearAdjust']=df['FiscalYearIssuance'].to_numpy()-df['FiscalYearIssuance'].mean()
 for v in vars:
  c=COL[v]
  if v in CATEG:
   s=df[c].astype('string').fillna('Unknown').astype(str)
   dum=pd.get_dummies(s,prefix=v,drop_first=True,dtype=float)
   X=pd.concat([X,dum],axis=1)
  else:
   z=df[c].astype(float)
   X[v]=(z-z.mean())/(z.std() or 1.)
 if interactions and with_p:
  z=df.logP.to_numpy()-df.logP.mean()
  for v in vars:
   cols=[x for x in X if x==v or x.startswith(v+'_')]
   for c in cols: X['P_X_'+c]=X[c].to_numpy()*z
 return sm.add_constant(X,has_constant='add').astype(float)

def nested_stats(df,component,v,adjust_year=False,do_interaction=False):
 y=np.log1p(df[COMP[component]].to_numpy())
 base=orthogonal_X(df,[],True,with_year=adjust_year)
 full=orthogonal_X(df,[v],True,with_year=adjust_year)
 mod0=sm.OLS(y,base).fit();mod1=sm.OLS(y,full).fit()
 rss0=np.sum(mod0.resid**2);rss1=np.sum(mod1.resid**2)
 pnew=full.shape[1]-base.shape[1]
 pr2=max(0.,(rss0-rss1)/rss0)
 if pnew<=0: p=np.nan
 else:
  fstat=max(0,(rss0-rss1)/pnew)/(rss1/max(mod1.df_resid,1))
  p=float(fdist.sf(fstat,pnew,mod1.df_resid))
 p_cluster=np.nan
 try:
  rob=sm.OLS(y,full).fit(cov_type='cluster',cov_kwds={'groups':df.IssuerName.to_numpy()})
  R=np.zeros((pnew,len(rob.params)))
  for j in range(pnew):R[j,len(base.columns)+j]=1
  p_cluster=float(rob.wald_test(R,scalar=True).pvalue)
 except Exception:pass
 if not do_interaction: return {'component':component,'variable':v,'partialR2':pr2,'p_f':p,'p_issuer_cluster':p_cluster,'df_added':pnew,'year_adjusted':adjust_year}
 full2=orthogonal_X(df,[v],True,with_year=adjust_year,interactions=True)
 pnew2=full2.shape[1]-full.shape[1]
 if pnew2<=0:return {'component':component,'variable':v,'interaction_p':np.nan,'interaction_partialR2':0.,'interaction_df':0}
 mod2=sm.OLS(y,full2).fit()
 rss2=np.sum(mod2.resid**2)
 if mod2.df_resid <=0: p2=np.nan
 else: p2=float(fdist.sf(max(0,(rss1-rss2)/pnew2)/(rss2/mod2.df_resid),pnew2,mod2.df_resid))
 try:
  rob=sm.OLS(y,full2).fit(cov_type='cluster',cov_kwds={'groups':df.IssuerName.to_numpy()})
  R=np.zeros((pnew2,len(rob.params)))
  for j in range(pnew2): R[j,len(full.columns)+j]=1
  pc=float(rob.wald_test(R,scalar=True).pvalue)
 except:pc=np.nan
 return {'component':component,'variable':v,'interaction_p':p2,'interaction_cluster_p':pc,'interaction_partialR2':max(0,(rss1-rss2)/rss1),'interaction_df':pnew2}

def dmodel(features,scale):
 transformations=[]
 if 'Principal Amount' in features:
  transformations.append(('principal',SplineTransformer(n_knots=4,degree=2,include_bias=False,knots='quantile'),['logP']))
 numeric=[COL[v] for v in features if v in NUM and v!='Principal Amount']
 categoricals=[COL[v] for v in features if v in CATEG]
 if numeric: transformations.append(('num',StandardScaler(),numeric))
 if categoricals:transformations.append(('categorical',OneHotEncoder(handle_unknown='ignore'),categoricals))
 if not transformations:return None
 return Pipeline([('transform',ColumnTransformer(transformations,sparse_threshold=0.0)),('est',Ridge(alpha=10.0))])

def cv_preds(df,component,features,scale,mode='additive',return_details=False):
 # Fixed and pre-declared diagnostic forms; no hyperparameter tuning here.
 actuals=[];preds=[]; yrs=[];inds=[]
 for year in [2018,2019,2020,2021]:
  tr=df[df.FiscalYearIssuance<year];te=df[df.FiscalYearIssuance==year]
  y=tr[COMP[component]].to_numpy(dtype=float)
  if features:
   m=dmodel(features,scale)
   if scale=='log1p':m.fit(tr,np.log1p(y));p=np.expm1(m.predict(te))
   else: m.fit(tr,y);p=m.predict(te)
   p=np.maximum(0,p)
  else:
   p=np.repeat(np.median(y),len(te))
  preds.extend(p);actuals.extend(te[COMP[component]].to_numpy());yrs.extend([year]*len(te));inds.extend(te.index.tolist())
 actuals=np.array(actuals);preds=np.array(preds);yrs=np.array(yrs)
 res={'component':component,'features':'+'.join(features) if features else '(none)','form':scale,'MAE':mean_absolute_error(actuals,preds),'n':len(actuals)}
 for yr in [2018,2019,2020,2021]:res['MAE_'+str(yr)]=mean_absolute_error(actuals[yrs==yr],preds[yrs==yr])
 if return_details: return res,pd.DataFrame({'index':inds,'year':yrs,'actual':actuals,'prediction':preds})
 return res

def improvement(df,component,baseline,candidate):
 # paired predictions, by year, using both dollar-scale and log-scale ridge diagnostic fits
 results=[]
 for scale in ['raw','log1p']:
  a=cv_preds(df,component,baseline,scale if baseline else 'median')
  b=cv_preds(df,component,candidate,scale)
  gains=[a['MAE_'+str(yr)]-b['MAE_'+str(yr)] for yr in [2018,2019,2020,2021]]
  results.append({'form':scale,'gain_usd':a['MAE']-b['MAE'],'gain_pct':100*(a['MAE']-b['MAE'])/a['MAE'],'positive_years':sum(v>0 for v in gains),'baseline_mae':a['MAE'],'candidate_mae':b['MAE'],**{'gain_'+str(yr):g for yr,g in zip([2018,2019,2020,2021],gains)}})
 return results


def interaction_cv(df,component,baseline,v,scale):
 from sklearn.linear_model import Ridge
 actuals=[];preds=[];years=[]
 if 'Principal Amount' not in baseline: return None
 for year in [2018,2019,2020,2021]:
  tr=df[df.FiscalYearIssuance<year];te=df[df.FiscalYearIssuance==year]
  sp=SplineTransformer(n_knots=4,degree=2,include_bias=False,knots='quantile')
  p_tr=sp.fit_transform(tr[['logP']]);p_te=sp.transform(te[['logP']])
  if v in CATEG:
   enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
   ztr=enc.fit_transform(tr[[COL[v]]]);zte=enc.transform(te[[COL[v]]])
  else:
   std=StandardScaler();ztr=std.fit_transform(tr[[COL[v]]]);zte=std.transform(te[[COL[v]]])
  mu=tr.logP.mean();sigma=tr.logP.std() or 1.
  xtr=(tr.logP.to_numpy()-mu)/sigma
  xte=(te.logP.to_numpy()-mu)/sigma
  Xtr=np.column_stack([p_tr,ztr,ztr*xtr[:,None]])
  Xte=np.column_stack([p_te,zte,zte*xte[:,None]])
  y=tr[COMP[component]].to_numpy(float)
  yfit=np.log1p(y) if scale=='log1p' else y
  m=Ridge(alpha=10.).fit(Xtr,yfit)
  pred=m.predict(Xte)
  if scale=='log1p':pred=np.expm1(pred)
  preds.extend(np.maximum(0,pred));actuals.extend(te[COMP[component]].to_numpy(float));years.extend([year]*len(te))
 actuals=np.array(actuals);preds=np.array(preds);years=np.array(years)
 res={'form':scale,'MAE':mean_absolute_error(actuals,preds)}
 for year in [2018,2019,2020,2021]:res['MAE_'+str(year)]=mean_absolute_error(actuals[years==year],preds[years==year])
 return res

def interaction_improvement(df,component,base,v):
 out=[]
 for scale in ['raw','log1p']:
  b=cv_preds(df,component,base,scale)
  z=interaction_cv(df,component,base,v,scale)
  if z is None:continue
  gains=[b['MAE_'+str(year)]-z['MAE_'+str(year)] for year in [2018,2019,2020,2021]]
  out.append({'form':scale,'gain_usd':b['MAE']-z['MAE'],'gain_pct':100*(b['MAE']-z['MAE'])/b['MAE'],'positive_years':sum(g>0 for g in gains),'baseline_mae':b['MAE'],'candidate_mae':z['MAE'],**{'gain_'+str(yr):g for yr,g in zip([2018,2019,2020,2021],gains)}})
 return out

