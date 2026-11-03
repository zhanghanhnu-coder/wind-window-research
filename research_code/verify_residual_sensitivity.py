"""Read-only verification of all alternate block designs and saved draws."""
from datetime import datetime,timezone
import numpy as np,pandas as pd
from model import ART
from research_residual_shape import read,sha,js
from verify_window_error_attribution import reference_bound,close

def verify():
    out=ART/'residual_warm_v1';assert read(out/'block_sensitivity_complete.json')['status']=='COMPLETE'
    for p,h in read(out/'block_sensitivity_identity.json').items():assert sha(p)==h,p
    folds=['M05','M07','M09','M11'];dates=['2024-05-01','2024-07-01','2024-09-01','2024-11-01'];seeds=[20241101,20241102,20241103];ts=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');pops={f:dict(np.load(out/f/'population.npz')) for f in folds};drawrng=np.random.default_rng(20261209);hour={};events={}
    for fold in folds:
        b=np.mean([np.load(out/fold/f'hourly_scores_B_{s}.npz')['crps'] for s in seeds],0);t=np.mean([np.load(out/fold/f'hourly_scores_WT_{s}.npz')['crps'] for s in seeds],0);hour[fold]=(np.nansum(t-b,(1,2)),np.isfinite(b).sum((1,2)))
    for kind in ['single','group']:
        y=np.concatenate([pops[f][kind+'_labels'] for f in folds]);keys=np.concatenate([pops[f][kind+'_keys'][pops[f][kind+'_inverse']].reshape(-1,2) for f in folds]);_,ix=np.unique(keys,axis=0,return_inverse=True);ix=ix.reshape(y.shape);high=np.concatenate([np.load(out/f/'fixed_cohorts.npz')[kind] for f in folds]);p={c:np.stack([np.concatenate([np.load(out/f/f'diagnostic_{c}_{s}.npz')[kind] for f in folds]) for s in seeds]) for c in ['B','WT']};c=np.mean(p['WT']**2-p['B']**2,0);d=2*np.mean(p['B']-p['WT'],0);events[kind]=(y,ix,high,c,d)
    rows=pd.read_csv(out/'block_sensitivity.csv');nbounds=0;ncrps=0
    for days in [3,7,14,28]:
        for shift in [0,.5]:
            name=f'd{days}_shift{int(shift*2)}';design=np.load(out/f'sensitivity_design_{name}.npz');bb=[];dd=[];offset=0;parts=[]
            for fold,date in zip(folds,dates):
                anchor=pd.Timestamp(date,tz='Asia/Shanghai')-pd.Timedelta(hours=days*24*shift);elapsed=(ts[pops[fold]['ids']].asi8-anchor.value)//3_600_000_000_000;b=elapsed//(days*24);_,b=np.unique(b,return_inverse=True);k=int(b.max())+1;draw=drawrng.integers(0,k,(2000,k));parts.append((b,draw));bb.append(b+offset);dd.append(draw+offset);offset+=k
            blocks=np.concatenate(bb);draws=np.concatenate(dd,1);close(design['blocks'],blocks,0);close(design['draws'],draws,0)
            if days==7 and shift==0:close(blocks,np.concatenate([pops[f]['blocks']+9*i for i,f in enumerate(folds)]),0)
            num=np.zeros(2000);den=np.zeros(2000);nonempty=[];pointnum=0.;pointden=0
            for fold,(b,d) in zip(folds,parts):
                s,n=hour[fold];ss=np.array([s[b==i].sum() for i in np.unique(b)]);nn=np.array([n[b==i].sum() for i in np.unique(b)]);num+=ss[d].sum(1);den+=nn[d].sum(1);nonempty.append(int((nn>0).sum()));pointnum+=s.sum();pointden+=n.sum()
            sample=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);saved=np.load(out/f'sensitivity_{name}_crps.npz');close(sample,saved['difference']);valid=np.isfinite(sample);supported=min(nonempty)>=4 and valid.sum()>=1900;row=rows[(rows.days==days)&(rows.anchor_shift_fraction==shift)&(rows.metric=='crps_all_observed')].iloc[0];assert bool(row.supported)==supported and row.minimum_nonempty_blocks==min(nonempty);close([row.point_lower,row.point_upper],[pointnum/pointden]*2)
            if supported:close([row.outer_low,row.outer_high],np.quantile(sample[valid],[.025,.975]))
            else:assert np.isnan(row.outer_low) and np.isnan(row.outer_high)
            ncrps+=len(sample)
            for kind in ['single','group']:
                y,ix,high,c,d=events[kind]
                for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high',high)]:
                    lo,hi,bl,bu,bn,_=reference_bound(c,d,y,ix,mask,blocks,draws);saved=np.load(out/f'sensitivity_{name}_{kind}_{cohort}.npz');close(saved['lower'],bl);close(saved['upper'],bu);close(saved['n'],bn);valid=np.isfinite(bl)&np.isfinite(bu);nonempty=[];start=0
                    for b,_ in parts:nonempty.append(int(sum(mask[start:start+len(b)][b==k].any() for k in np.unique(b))));start+=len(b)
                    supported=min(nonempty)>=4 and valid.sum()>=1900;row=rows[(rows.days==days)&(rows.anchor_shift_fraction==shift)&(rows.metric==kind+'_'+cohort)].iloc[0];assert bool(row.supported)==supported and row.minimum_nonempty_blocks==min(nonempty);close([row.point_lower,row.point_upper],[lo,hi])
                    if supported:close([row.outer_low,row.outer_high],[np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)])
                    else:assert np.isnan(row.outer_low) and np.isnan(row.outer_high)
                    nbounds+=len(bl)
            print('SENSITIVITY VERIFIED',days,shift,flush=True)
    arc=out/'sensitivity_initial_attempt';matched=0
    for original,info in read(arc/'archive_map.json').items():
        from pathlib import Path
        archived=Path(info['archived']);assert sha(archived)==info['sha256']
        if archived.suffix=='.npz':
            old=np.load(archived);new=np.load(original);assert old.files==new.files
            for key in old.files:close(old[key],new[key],0)
            matched+=1
        elif archived.name=='block_sensitivity.csv':pd.testing.assert_frame_equal(pd.read_csv(archived),pd.read_csv(original));matched+=1
    result=dict(status='PASS',settings=8,rows=40,supported_rows=int(rows.supported.sum()),independent_window_draws=nbounds,independent_crps_draws=ncrps,initial_attempt_numerical_files_unchanged=matched,utc=datetime.now(timezone.utc).isoformat());js(out/'block_sensitivity_verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()
