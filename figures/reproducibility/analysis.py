from pathlib import Path
import json,itertools,platform
import numpy as np
import pandas as pd
import pvlib,sklearn,xgboost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import reference as ps
OUT=Path('results');OUT.mkdir(exist_ok=True)
DAILY=['GHI_d','T_mean','RH_mean','WS_mean','P_total','doy_sin','doy_cos']
EXTRA=['H_poa','M2','MT','MW','closure_signed','closure_abs']
GRID=[dict(n_estimators=n,max_depth=d,learning_rate=l) for n,d,l in itertools.product([200,500],[3,5,7],[.03,.1])]
def model(p):return XGBRegressor(**p,subsample=.8,random_state=42,n_jobs=2)
def metrics(y,p):
 e=np.asarray(p)-np.asarray(y)
 return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),ME=float(e.mean()),R2=float(1-np.sum(e**2)/np.sum((y-np.mean(y))**2)))
def features(cfg):
 raw=ps.load_weather();d=ps.simulate(cfg,raw)
 w=raw.rename(columns={'ALLSKY_SFC_SW_DWN':'ghi','ALLSKY_SFC_SW_DNI':'dni','ALLSKY_SFC_SW_DIFF':'dhi','T2M':'temp_air','WS10M':'wind_speed'}).copy();w.index+=pd.Timedelta(minutes=30)
 sol=ps.location.get_solarposition(w.index);cosz=np.cos(np.deg2rad(sol.zenith)).clip(lower=0)
 if cfg['irr']=='closure_dni':w['dni']=np.where(cosz>.1,((w.ghi-w.dhi)/cosz.replace(0,np.nan)).clip(0,1400),0.)
 arr=pvlib.pvsystem.Array(pvlib.pvsystem.FixedMount(cfg['tilt'],cfg['az']),albedo=cfg['albedo'],module_parameters={'pdc0':1000,'gamma_pdc':cfg['gamma']},temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'])
 sys=pvlib.pvsystem.PVSystem(arrays=[arr],inverter_parameters={'pdc0':1000/cfg['dcac'],'eta_inv_nom':cfg['eta']},losses_parameters=cfg['loss'])
 mc=pvlib.modelchain.ModelChain.with_pvwatts(sys,ps.location,aoi_model='physical',spectral_model='no_loss',temperature_model='sapm');mc.prepare_inputs(w[['ghi','dni','dhi','temp_air','wind_speed']]);poa=mc.results.total_irrad.poa_global
 q=w.ghi-w.dhi-w.dni*cosz
 h=pd.DataFrame({'M2':(poa/1000)**2,'MT':poa*(w.temp_air-25)/1000,'MW':poa*w.wind_speed/1000,'closure_signed':q/1000,'closure_abs':np.abs(q)/1000});h.index=(raw.index+pd.Timedelta(hours=8)).tz_localize(None)
 d=d.join(h.groupby(h.index.normalize()).sum());assert np.isfinite(d.to_numpy()).all();assert len(d)==1825
 return d

def predict(d,name,params,trmask,temask):
 tr=d.loc[trmask];te=d.loc[temask]
 if name=='Calendar':return tr.groupby(tr.index.month).E.mean().reindex(te.index.month).to_numpy()
 if name=='GHI ratio':return tr.E.sum()/tr.GHI_d.sum()*te.GHI_d.to_numpy()
 if name=='POA ratio':return tr.E.sum()/tr.H_poa.sum()*te.H_poa.to_numpy()
 if name=='Daily linear':return LinearRegression().fit(tr[DAILY],tr.E).predict(te[DAILY])
 if name=='Daily RF':return RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=2,random_state=42).fit(tr[DAILY],tr.E).predict(te[DAILY])
 cols=DAILY if name=='Daily XGB' else DAILY+(['H_poa'] if name=='Residual POA' else (EXTRA[:4] if name in ['Residual moments','Expanded moments'] else EXTRA))
 residual=name.startswith('Residual');k=tr.E.sum()/tr.H_poa.sum() if residual else 0
 m=model(params).fit(tr[cols],tr.E-(k*tr.H_poa if residual else 0));p=m.predict(te[cols])+(k*te.H_poa.to_numpy() if residual else 0)
 return np.clip(p,0,24*.96/1.1) if residual else p

def run():
 d=features(ps.BASE);d.to_csv(OUT/'daily-features.csv',index_label='date')
 archive=Path('data/archived-daily-reference.csv')
 archived=pd.read_csv(archive,index_col=0)
 assert np.allclose(d.E,archived.E,atol=1e-9)
 a=d.index.year<=2023;v=d.index.year==2024;t=d.index.year==2025
 methods=['Calendar','GHI ratio','Daily linear','Daily RF','Daily XGB','POA ratio','Expanded XGB','Expanded moments','Residual POA','Residual moments','Residual full'];best={};tuning=[]
 for name in ['Daily XGB','Expanded XGB','Expanded moments','Residual POA','Residual moments','Residual full']:
  scores=[]
  for p in GRID:
   r=metrics(d.loc[v].E,predict(d,name,p,a,v));scores.append(r['RMSE']);tuning.append(dict(method=name,**p,**r))
  best[name]=GRID[int(np.argmin(scores))];print('selected',name,best[name],min(scores),flush=True)
 pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
 y=d.loc[t].E.to_numpy();pred={n:predict(d,n,best.get(n,{}),~t,t) for n in methods}
 pd.DataFrame({'reference':y,**pred},index=d.loc[t].index).to_csv(OUT/'predictions.csv',index_label='date')
 rng=np.random.default_rng(42);months=d.loc[t].index.month.to_numpy();blocks=[np.where(months==i)[0] for i in range(1,13)];samples=[np.concatenate([blocks[j] for j in rng.integers(0,12,12)]) for _ in range(2000)];rows=[]
 for n,p in pred.items():
  lo,hi=np.quantile([np.mean(np.abs(p[i]-y[i])) for i in samples],[.025,.975]);rows.append(dict(method=n,**metrics(y,p),MAE_lo=lo,MAE_hi=hi))
 perf=pd.DataFrame(rows);perf.to_csv(OUT/'performance.csv',index=False);print(perf.to_string(index=False),flush=True)
 paired=[]
 for n in ['Daily XGB','POA ratio','Expanded XGB','Expanded moments','Residual POA']:
  delta=np.abs(pred['Residual moments']-y)-np.abs(pred[n]-y);ci=np.quantile([delta[i].mean() for i in samples],[.025,.975]);paired.append(dict(comparator=n,difference=delta.mean(),lo=ci[0],hi=ci[1]))
 pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False)
 hist=[]
 for year in [2022,2023,2024,2025]:
  for n in ['Daily XGB','POA ratio','Expanded XGB','Expanded moments','Residual POA','Residual moments','Residual full']:
   pp=predict(d,n,best.get(n,{}),d.index.year<year,d.index.year==year);hist.append(dict(year=year,method=n,**metrics(d.loc[d.index.year==year].E,pp)))
 pd.DataFrame(hist).to_csv(OUT/'historical.csv',index=False)
 thresholds=d.loc[~t].closure_abs.quantile([1/3,2/3]).to_numpy();test=d.loc[t];groups=[]
 for label,mask in {'low closure':test.closure_abs<=thresholds[0],'mid closure':(test.closure_abs>thresholds[0])&(test.closure_abs<=thresholds[1]),'high closure':test.closure_abs>thresholds[1],'rain >=10 mm':test.P_total>=10,'hot > train q90':test.T_mean>d.loc[~t].T_mean.quantile(.9)}.items():
  for n in ['Daily XGB','POA ratio','Residual moments','Residual full']:groups.append(dict(group=label,method=n,n=int(mask.sum()),**metrics(y[mask],pred[n][mask])))
 pd.DataFrame(groups).to_csv(OUT/'regimes.csv',index=False)
 dc=features({**ps.BASE,'irr':'closure_dni'});sens=[]
 for dd,label in [(d,'Downloaded components'),(dc,'Algebraic DNI intervention')]:
  for n in ['Daily XGB','POA ratio','Residual moments','Residual full']:
   pp=predict(dd,n,best.get(n,{}),~t,t);sens.append(dict(reference=label,method=n,annual=dd.loc[t].E.sum(),**metrics(dd.loc[t].E,pp)))
 pd.DataFrame(sens).to_csv(OUT/'intervention.csv',index=False)
 meta={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'pvlib':pvlib.__version__,'xgboost':xgboost.__version__,'best':best,'n_days':len(d),'split_counts':[int(a.sum()),int(v.sum()),int(t.sum())],'closure_thresholds':thresholds.tolist(),'max_reference_difference':float(np.max(np.abs(d.E-archived.E.to_numpy())))}
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':run()
