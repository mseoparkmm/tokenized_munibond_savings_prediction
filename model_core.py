"""Model family comparison AFTER input selection, historical annual tests, 465-row refit.

Inputs and principal interactions are imported from frozen selection JSON.
No model-family score influences predictor selection.
"""
from pathlib import Path
import os,json,joblib,warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, Ridge, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from data_and_diagnostics import COMP, COL, NUM, CATEG, GATE
warnings.filterwarnings('ignore', category=FutureWarning)
ROOT=Path(__file__).resolve().parent
CV_YEARS=(2018,2019,2020,2021)
BACKTEST_YEARS=(2022,2023,2024,2025)

class Design:
 def __init__(self,features,interactions,variant):
  self.features=list(features);self.interactions=list(interactions);self.variant=variant
  assert set(self.interactions).issubset(self.features)
 def fit(self,df):
  if 'Principal Amount' in self.features:
   if self.variant=='spline':
    self.principal=SplineTransformer(n_knots=4,degree=2,include_bias=False,knots='quantile')
    self.principal.fit(df[['logP']]);self.p_col='logP'
   else:
    self.principal=StandardScaler();self.p_col='ActualPar' if self.variant=='raw' else 'logP'
    self.principal.fit(df[[self.p_col]])
  self.z={}
  for v in self.features:
   if v=='Principal Amount':continue
   if v in CATEG:self.z[v]=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
   else:self.z[v]=StandardScaler()
   self.z[v].fit(df[[COL[v]]])
  self.p_mean=float(df.logP.mean());self.p_std=float(df.logP.std()) or 1.
  return self
 def transform(self,df,exclude_principal=False):
  vals=[];zs={}
  if 'Principal Amount' in self.features and not exclude_principal:
   vals.append(self.principal.transform(df[[self.p_col]]))
  for v in self.features:
   if v=='Principal Amount':continue
   zs[v]=self.z[v].transform(df[[COL[v]]]);vals.append(zs[v])
  if self.interactions:
   logpc=((df.logP.to_numpy(float)-self.p_mean)/self.p_std)
   for v in self.interactions:vals.append(zs[v]*logpc[:,None])
  if not vals:return np.empty((len(df),0),float)
  return np.column_stack(vals)

class FixedPower:
 def __init__(self,features,interactions):
  self.features=list(features);self.interactions=list(interactions)
 def fit(self,df,y):
  self.design=Design(self.features,self.interactions,'raw').fit(df)
  z=self.design.transform(df,exclude_principal=True)
  p=df.ActualPar.to_numpy(float)/1e6
  med=float(np.median(y));base=max(1.,med/(max(np.median(p),1.)**.6))
  st=np.r_[max(0.,med*.2),base,.6,np.zeros(z.shape[1])]
  lo=np.r_[0.,0.,.05,np.full(z.shape[1],-np.inf)]
  hi=np.r_[1e9,1e9,2.5,np.full(z.shape[1],np.inf)]
  f=lambda a:a[0]+a[1]*p**a[2]+z@a[3:]-y
  fit=least_squares(f,st,bounds=(lo,hi),loss='soft_l1',f_scale=max(1000.,med*.25),max_nfev=250)
  self.params=fit.x
  return self
 def predict(self,df):
  z=self.design.transform(df,exclude_principal=True)
  a=self.params;p=df.ActualPar.to_numpy(float)/1e6
  return np.maximum(0,a[0]+a[1]*p**a[2]+z@a[3:])

class Estimator:
 def __init__(self,component,features,interactions,model_name):
  self.component=component;self.features=list(features);self.interactions=list(interactions);self.model_name=model_name
 def fit(self,df):
  y=df[COMP[self.component]].to_numpy(float)
  if self.model_name=='Median':
   self.median=float(np.median(y));return self
  if self.model_name=='FixedPower':
   self.model=FixedPower(self.features,self.interactions).fit(df,y);return self
  if self.model_name.startswith('CatBoost'):
   self.cols=[COL[v] for v in self.features]
   self.cats=[COL[v] for v in self.features if v in CATEG]
   xx=df[self.cols].copy()
   for col in self.cats:xx[col]=xx[col].fillna('Unknown').astype(str)
   yy=y.copy()
   if self.model_name=='CatBoost Log':yy=np.log1p(y)
   if self.model_name=='CatBoost Ratio':yy=y/(df.ActualPar.to_numpy(float)/1e6)
   loss='RMSE' if self.model_name=='CatBoost Log' else 'MAE'
   self.model=CatBoostRegressor(iterations=300,depth=3,learning_rate=.04,l2_leaf_reg=12,
                 loss_function=loss,verbose=False,allow_writing_files=False,random_seed=42,thread_count=2)
   self.model.fit(xx,yy,cat_features=self.cats)
   return self
  method=self.model_name
  variant='spline' if method.startswith('Spline') else ('raw' if method in ('OLS Raw','Ridge Raw','Quantile Raw') else 'log')
  self.design=Design(self.features,self.interactions,variant).fit(df)
  xx=self.design.transform(df)
  if method.startswith('OLS'):reg=LinearRegression()
  elif method.startswith('Quantile'):reg=QuantileRegressor(quantile=.5,alpha=.02,solver='highs')
  else:reg=Ridge(alpha=10.)
  yy=np.log1p(y) if method in ('OLS Log','Ridge Log','SplineRidge Log') else y
  reg.fit(xx,yy)
  self.model=reg
  return self
 def predict(self,df):
  if self.model_name=='Median':return np.repeat(self.median,len(df))
  if self.model_name=='FixedPower':return self.model.predict(df)
  if self.model_name.startswith('CatBoost'):
   xx=df[self.cols].copy()
   for col in self.cats:xx[col]=xx[col].fillna('Unknown').astype(str)
   p=self.model.predict(xx)
   if self.model_name=='CatBoost Log':p=np.expm1(p)
   if self.model_name=='CatBoost Ratio':p=p*df.ActualPar.to_numpy(float)/1e6
  else:
   p=self.model.predict(self.design.transform(df))
   if self.model_name in ('OLS Log','Ridge Log','SplineRidge Log'):p=np.expm1(p)
  return np.maximum(0.,np.asarray(p,dtype=float))


def model_candidates(features):
 if not features:return ['Median']
 return ['Median','OLS Raw','OLS Log','Ridge Raw','Ridge Log','Quantile Raw',
         'SplineRidge Raw','SplineRidge Log','FixedPower','CatBoost Raw','CatBoost Log','CatBoost Ratio']


def score(y,p):
 a=np.asarray(y,float);p=np.asarray(p,float)
 result={'N':len(a),'MAE':float(mean_absolute_error(a,p)),
 'RMSE':float(np.sqrt(mean_squared_error(a,p))),'R2':float(r2_score(a,p))}
 result['MAPE_percent']=float(100*np.mean(np.abs(a-p)/a)) if np.all(a>0) else np.nan
 return result


