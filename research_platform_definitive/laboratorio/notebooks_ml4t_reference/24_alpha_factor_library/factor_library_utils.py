"""Modern alpha factor library helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def ensure_output_dir(path: str | Path = "../data/alpha_factor_library") -> Path:
    output=Path(path); output.mkdir(parents=True, exist_ok=True); return output


def make_ohlcv(n_assets:int=40, periods:int=500, seed:int=24)->pd.DataFrame:
    rng=np.random.default_rng(seed); dates=pd.bdate_range(end=pd.Timestamp('2026-05-08'), periods=periods); rows=[]
    for i in range(n_assets):
        r=rng.normal(.0002,.012,periods); close=50*np.exp(np.cumsum(r)); open_=close*(1+rng.normal(0,.002,periods)); high=np.maximum(open_,close)*(1+rng.random(periods)*.01); low=np.minimum(open_,close)*(1-rng.random(periods)*.01); vol=rng.integers(100_000,2_000_000,periods)
        rows.append(pd.DataFrame({'date':dates,'ticker':f'A{i:03d}','open':open_,'high':high,'low':low,'close':close,'volume':vol}))
    return pd.concat(rows,ignore_index=True)


def add_factors(df:pd.DataFrame)->pd.DataFrame:
    out=df.sort_values(['ticker','date']).copy(); g=out.groupby('ticker')
    out['return_1d']=g.close.pct_change(); out['momentum_21']=g.close.pct_change(21); out['momentum_63']=g.close.pct_change(63)
    out['sma_20']=g.close.transform(lambda s:s.rolling(20).mean()); out['sma_ratio']=out.close/out.sma_20-1
    out['volatility_21']=g.return_1d.transform(lambda s:s.rolling(21).std()); out['dollar_volume']=out.close*out.volume
    out['liquidity_rank']=out.groupby('date').dollar_volume.rank(pct=True); out['reversal_5']=-g.close.pct_change(5)
    out['alpha_001']=out.reversal_5.rank(pct=True); out['alpha_002']=out.momentum_21.rank(pct=True)-out.volatility_21.rank(pct=True); out['alpha_003']=out.sma_ratio.rank(pct=True)
    out['forward_return_5d']=g.close.pct_change(5).shift(-5)
    return out.dropna().reset_index(drop=True)


def indicator_zoo()->pd.DataFrame:
    return pd.DataFrame({'category':['momentum','trend','volatility','liquidity','formulaic'],'factor':['momentum_21','sma_ratio','volatility_21','liquidity_rank','alpha_001/002/003'],'description':['21-day return','close vs 20-day average','21-day realized volatility','cross-sectional dollar-volume rank','compact formulaic alpha examples']})


def evaluate_factors(panel:pd.DataFrame)->pd.DataFrame:
    factors=['momentum_21','momentum_63','sma_ratio','volatility_21','liquidity_rank','reversal_5','alpha_001','alpha_002','alpha_003']
    rows=[]
    for f in factors:
        rows.append({'factor':f,'ic':panel[f].corr(panel.forward_return_5d, method='spearman')})
    return pd.DataFrame(rows).sort_values('ic', key=lambda s:s.abs(), ascending=False)


def model_importance(panel:pd.DataFrame)->pd.DataFrame:
    factors=['momentum_21','momentum_63','sma_ratio','volatility_21','liquidity_rank','reversal_5','alpha_001','alpha_002','alpha_003']
    train,test=train_test_split(panel, test_size=.3, random_state=42, shuffle=False)
    model=RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1).fit(train[factors], train.forward_return_5d)
    pred=model.predict(test[factors]); imp=pd.DataFrame({'factor':factors,'importance':model.feature_importances_}).sort_values('importance', ascending=False)
    return imp.assign(test_rmse=float(np.sqrt(mean_squared_error(test.forward_return_5d,pred))))


def quantile_returns(panel:pd.DataFrame, factor:str='alpha_002')->pd.DataFrame:
    frame=panel.copy(); frame['quantile']=frame.groupby('date')[factor].transform(lambda s:pd.qcut(s.rank(method='first'),5,labels=False)+1)
    return frame.groupby('quantile').forward_return_5d.agg(['mean','std','count']).reset_index()
