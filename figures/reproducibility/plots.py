from pathlib import Path
from io import BytesIO
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon
from matplotlib.colors import TwoSlopeNorm
OUT=Path('..');R=Path('results')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14,'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none'})
NAVY='#17324d';TEAL='#007f86';GOLD='#d69b24';RED='#ba4d4d';GRAY='#687a8a';BLUE='#4278ae'
COL={'Daily XGB':BLUE,'POA ratio':GOLD,'Expanded moments':GRAY,'Residual POA':'#8f6bb0','Residual moments':TEAL,'Residual full':RED}
LABEL={'Daily XGB':'Daily XGBoost','POA ratio':'POA ratio','Expanded moments':'Direct / moments','Residual POA':'Residual / POA','Residual moments':'Bounded residual / moments','Residual full':'Residual / moments + closure','Expanded XGB':'Direct / moments + closure'}
def save(fig,name):
 for ext in ['pdf','svg','png']:
  buffer=BytesIO();fig.savefig(buffer,format=ext,dpi=320,metadata={'Creator':'Research figures'} if ext=='pdf' else None)
  (OUT/(name+'.'+ext)).write_bytes(buffer.getvalue())
 plt.close(fig)
def panel(ax,s):ax.text(0,1.03,s,transform=ax.transAxes,fontweight='bold',color=NAVY,fontsize=11)
def box(ax,x,y,w,h,title,body,fc='#edf4f7'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.02',facecolor=fc,edgecolor='#adc0cc',linewidth=1))
 ax.text(x+w/2,y+h*.70,title,ha='center',va='center',fontweight='bold',color=NAVY,fontsize=10)
 ax.text(x+w/2,y+h*.32,body,ha='center',va='center',color=NAVY,fontsize=8.5,linespacing=1.5)
def arrow(ax,start,end,text=None,color=NAVY):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.3,connectionstyle='arc3,rad=0'))
 if text:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.018,text,ha='center',fontsize=8,color=color,bbox={'facecolor':'white','edgecolor':'none','pad':1.0})
# Architecture: target branch is separate from predictor branch.
fig,ax=plt.subplots(figsize=(10.5,6.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.77,.26,.18,'Hourly weather archive','NASA POWER | 2021–2025\nGHI, DNI, DHI, temperature, wind')
box(ax,.37,.77,.26,.18,'Solar geometry + POA','Midpoint solar position\nPerez transposition')
box(ax,.72,.77,.26,.18,'Reference branch','Optics → temperature → DC → AC\nDaily simulated energy $E_d$')
arrow(ax,(.29,.86),(.36,.86));arrow(ax,(.64,.86),(.71,.86))
box(ax,.03,.42,.28,.20,'Daily predictor block','Weather summaries + calendar\nPOA and thermal–wind moments')
box(ax,.37,.42,.26,.20,'Physical anchor','$b_d=\\hat{k}H_d$\nFit coefficient on earlier years',fc='#fff5dd')
box(ax,.71,.42,.27,.20,'Residual tree ensemble','$r_d=E_d-b_d$\nRegularized XGBoost',fc='#e2f3ee')
arrow(ax,(.5,.76),(.17,.63));arrow(ax,(.31,.52),(.36,.52));arrow(ax,(.64,.52),(.70,.52));arrow(ax,(.85,.76),(.85,.63),text='training target')
box(ax,.36,.08,.30,.19,'Bounded reconstruction','$\\hat E_d=\\Pi_{[0,U]}(b_d+g(\\mathbf{z}_d))$\nNo target used during prediction',fc='#e2f3ee')
arrow(ax,(.50,.41),(.50,.28));arrow(ax,(.85,.41),(.67,.19))
ax.text(.03,.15,'Audit branch\nClosure discrepancy is diagnostic;\nadding it is tested as an ablation.',fontsize=9,color=GRAY,va='center')
save(fig,'architecture')
# Algorithm flowchart, explicit validation loop.
fig,ax=plt.subplots(figsize=(8.5,7.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.26,.86,.48,.10,'Audit input archive','Validate timestamps, units and complete local days')
box(ax,.26,.70,.48,.10,'Chronological partition','2021–2023 fit | 2024 select | 2025 evaluate')
box(ax,.26,.54,.48,.10,'For each candidate setting','Fit anchor + residual trees using training data')
for y in [.86,.70]:arrow(ax,(.5,y-.01),(.5,y-.055))

ax.add_patch(Polygon([[.5,.48],[.74,.40],[.5,.32],[.26,.40]],closed=True,fc='#fff5dd',ec=GOLD,lw=1.2));ax.text(.5,.40,'All 12 settings evaluated\non 2024?',ha='center',va='center',fontsize=9,fontweight='bold',color=NAVY)
arrow(ax,(.5,.53),(.5,.49));arrow(ax,(.26,.4),(.1,.4));arrow(ax,(.1,.4),(.1,.59));arrow(ax,(.1,.59),(.25,.59));ax.text(.13,.43,'No',color=GRAY)
box(ax,.26,.17,.48,.10,'Refit selected configuration','Use 2021–2024; freeze coefficients and trees')
arrow(ax,(.5,.31),(.5,.28));ax.text(.53,.29,'Yes',fontsize=9,color=TEAL)
box(ax,.26,.01,.48,.10,'Evaluate and audit 2025','Bounds, paired bootstrap, regimes and ablations')
arrow(ax,(.5,.16),(.5,.12));save(fig,'algorithm-flowchart')
perf=pd.read_csv(R/'performance.csv').set_index('method');pred=pd.read_csv(R/'predictions.csv',index_col=0,parse_dates=True);data=pd.read_csv(R/'daily-features.csv',index_col=0,parse_dates=True);test=data.loc['2025'];y=pred.reference
# Performance interval + tail comparison
names=['Daily XGB','POA ratio','Expanded moments','Residual POA','Residual moments','Residual full']
fig,axs=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[1.15,1]})
for i,n in enumerate(names):
 r=perf.loc[n];axs[0].errorbar(r.MAE,i,xerr=[[r.MAE-r.MAE_lo],[r.MAE_hi-r.MAE]],fmt='o',color=COL[n],capsize=3,ms=7)
axs[0].set(yticks=range(len(names)),yticklabels=[LABEL[n] for n in names],xscale='log',xlabel='MAE [kWh/day per kWp]');axs[0].invert_yaxis();panel(axs[0],'a  Mean error and 95% month-block intervals')
for n in ['Daily XGB','POA ratio','Residual moments']:
 err=np.sort(np.abs(pred[n]-y));axs[1].plot(err,np.arange(1,len(err)+1)/len(err),color=COL[n],lw=2,label=LABEL[n])
axs[1].set(xlim=(0,.55),ylim=(0,1.01),xlabel='Absolute error [kWh/day per kWp]',ylabel='Fraction of test days');axs[1].legend(fontsize=8,loc='lower right');panel(axs[1],'b  Distribution of daily approximation errors');fig.tight_layout(w_pad=2);save(fig,'performance')
# Error atlas: calendar heatmaps with common scale and residuals.
fig,axs=plt.subplots(2,2,figsize=(10.7,7.0));norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
for ax,n,tag in [(axs[0,0],'Daily XGB','a'),(axs[0,1],'Residual moments','b')]:
 a=np.full((12,31),np.nan)
 for date,value in (pred[n]-y).items():a[date.month-1,date.day-1]=value
 im=ax.imshow(a,aspect='auto',cmap='RdBu_r',norm=norm);ax.set(yticks=range(12),yticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],xticks=[0,9,19,30],xticklabels=[1,10,20,31],xlabel='Day of month');ax.grid(False);panel(ax,tag+'  '+LABEL[n])

for n in ['Daily XGB','POA ratio','Residual moments']:
 axs[1,0].scatter(test.closure_abs,pred[n]-y,s=11,alpha=.5,color=COL[n],label=LABEL[n])
 axs[1,1].scatter(y,pred[n]-y,s=11,alpha=.5,color=COL[n])
for ax in axs[1]:ax.axhline(0,color=NAVY,lw=.9);ax.set_ylabel('Signed error [kWh/day per kWp]')
axs[1,0].set_xlabel('Daily absolute closure discrepancy [kWh/m²]');axs[1,1].set_xlabel('Simulated daily energy [kWh/day per kWp]');panel(axs[1,0],'c  Discrepancy and residual structure');panel(axs[1,1],'d  High-energy behavior');axs[1,0].legend(fontsize=7)
fig.subplots_adjust(left=.08,right=.89,wspace=.38,hspace=.45)
cax=fig.add_axes([.93,.57,.017,.31]);fig.colorbar(im,cax=cax,label='Prediction − simulation [kWh/day per kWp]')
save(fig,'error-atlas')
# Historical diagnostics and matched-input ablation.
h=pd.read_csv(R/'historical.csv');fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
for n in ['Daily XGB','POA ratio','Expanded moments','Residual moments']:
 z=h[h.method==n];axs[0].plot(z.year,z.MAE,'o-',color=COL[n],label=LABEL[n],lw=2)
axs[0].set(yscale='log',xticks=[2022,2023,2024,2025],xlabel='Evaluation year',ylabel='MAE [kWh/day per kWp]');axs[0].legend(fontsize=7);panel(axs[0],'a  Retrospective expanding-window diagnostics')
for i,n in enumerate(['Daily XGB','Expanded moments','Residual POA','Residual moments','Residual full']):
 axs[1].barh(i,perf.loc[n,'MAE'],color=COL[n],height=.65);axs[1].text(perf.loc[n,'MAE']*1.1,i,f"{perf.loc[n,'MAE']:.4f}",va='center',fontsize=8)
axs[1].set(xscale='log',xlim=(.005,.5),yticks=range(5),yticklabels=['Daily inputs / direct','Moments / direct','POA / residual','Moments / residual','+ closure / residual'],xlabel='2025 MAE [kWh/day per kWp]');axs[1].invert_yaxis();panel(axs[1],'b  Input and residual-learning ablations');fig.tight_layout(w_pad=2);save(fig,'ablation')
# Closure intervention and regime evaluation.
s=pd.read_csv(R/'intervention.csv');g=pd.read_csv(R/'regimes.csv');fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
for n in ['Daily XGB','POA ratio','Residual moments']:
 z=s[s.method==n];axs[0].plot([0,1],z.MAE,'o-',color=COL[n],lw=2,label=LABEL[n])
axs[0].set(xticks=[0,1],xticklabels=['Downloaded\ncomponents','Algebraic DNI\nintervention'],yscale='log',ylabel='MAE [kWh/day per kWp]');axs[0].legend(fontsize=7);panel(axs[0],'a  Target-changing sensitivity intervention')
for i,n in enumerate(['Daily XGB','POA ratio','Residual moments']):
 z=g[g.method==n];axs[1].plot(np.arange(len(z)),z.MAE,'o-',color=COL[n],label=LABEL[n])
axs[1].set(xticks=range(5),xticklabels=['Low\nclosure','Mid\nclosure','High\nclosure','Rain\n≥10 mm','Hot\ndays'],yscale='log',ylabel='MAE [kWh/day per kWp]');panel(axs[1],'b  Training-defined 2025 regimes');fig.tight_layout(w_pad=2);save(fig,'sensitivity')
# Reference distribution and time split
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8));year=data.index.year
for yy,c in zip(range(2021,2026),['#d2dce5','#b8cedb','#91b7c6',GOLD,TEAL]):
 z=data[year==yy];axs[0].plot(z.index,z.E,lw=.5,alpha=.65,color=c)
axs[0].axvspan(pd.Timestamp('2024-01-01'),pd.Timestamp('2024-12-31'),alpha=.1,color=GOLD);axs[0].axvspan(pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'),alpha=.09,color=TEAL);axs[0].set(ylabel='Simulated energy [kWh/day per kWp]');panel(axs[0],'a  Training, selection and evaluation record');axs[0].tick_params(axis='x',rotation=30)
ratio=data.H_poa/data.GHI_d
for m in range(1,13):
 z=ratio[data.index.month==m];axs[1].boxplot(z,positions=[m],widths=.55,showfliers=False,patch_artist=True,boxprops={'facecolor':'#d1e9e8','edgecolor':TEAL},medianprops={'color':TEAL})
axs[1].axhline(1,color=GOLD,ls='--');axs[1].set(xlabel='Month',ylabel='POA / horizontal daily irradiation',xticks=range(1,13));panel(axs[1],'b  Geometry information absent from daily GHI');fig.tight_layout(w_pad=2);save(fig,'reference-record')
