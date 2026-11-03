"""Exact cross-execution frozen-vs-rolling source pair; float64 evaluation only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,window_labels
from review_baselines_core import window_probabilities
from missing_window_bounds_core import physical_mapping
from state_gate_core import decision_counts
from lead_calibration_core import apply_groups

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';OUT=ART/'review_followup_v1/fixed_rolling'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
METHODS=['fixed_B','rolling28'];SEED=20261106;OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def event_losses(zero,one,y,ix):
    count=np.bincount(ix.ravel());z=np.bincount(ix.ravel(),weights=zero.ravel());o=np.bincount(ix.ravel(),weights=one.ravel());label=np.full(len(count),np.nan);known=np.isfinite(y);label[ix[known]]=y[known]
    np.testing.assert_array_equal(label[ix][known],y[known]);assert np.array_equal(np.isnan(label[ix]),np.isnan(y))
    return np.where(label==0,z,np.where(label==1,o,np.minimum(z,o))),np.where(label==0,z,np.where(label==1,o,np.maximum(z,o))),count

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope='existing cross-execution exact fixed_B/frozen versus rolling28; both legacy float32-trained/calibrated predictions scored with original float64 truth',mapping=dict(fixed_B='seasonal_rolling_v1 inventory method=frozen, source seasonal_probability_v1 calibrated_gated_graph_atoms',rolling28='seasonal_rolling_v1 inventory method=rolling28'),not_same_as_precision_pair='state_gate B is rolling28; state_gate W28 is residual_warm WT, not this fixed/rolling pair',bootstrap_seed=SEED,replicates=2000,quantiles=[.025,.975],threshold=10,decision_threshold=.8,main_direction='rolling28 minus fixed_B',folds=FOLDS,offsets=OFFSETS,unknowns='physical-window shared binary truth; denominator all forecast records; seed losses first averaged',resampling='fold-stratified target-start seven-day blocks anchored to first target date; retain all terminal partial blocks',no_training=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'preserve existing fixed protocol'
    else:js(OUT/'protocol.json',protocol)
    inventory=read(ART/'seasonal_rolling_v1/inventory.json');cross=read(ART/'review_followup_v1/cross_execution/source_identity.json');cp=read(ART/'review_followup_v1/cross_execution/protocol.json')
    assert cp['model']=='gated_graph_atoms' and cp['methods']=={'B':'frozen','W28':'rolling28'}
    times,nodes,raw=load_truth();rng=np.random.default_rng(SEED);identities=[];data={};identity={};context_checks=[]
    for f in FOLDS:
        fold=np.load(ART/'seasonal_probability_v1'/f/'data.npz');ids=fold['test_ids'];folder=OUT/f;folder.mkdir(exist_ok=True)
        ys=np.concatenate([window_labels(raw,ids,off) for off in OFFSETS],1);yg=np.stack([window_labels(raw,ids,off,node_indices=g,joint=True) for off in OFFSETS for g in GROUPS],1)
        pop=dict(ids=ids,single_labels=ys,group_labels=yg);prob={m:{k:[] for k in ['single','group']} for m in METHODS}
        for method,source_method in [('fixed_B','frozen'),('rolling28','rolling28')]:
            for seed in SEEDS:
                candidates=[r for r in inventory if (r['fold'],r['seed'],r['model'],r['method'])==(f,seed,'gated_graph_atoms',source_method)];assert len(candidates)==1;item=candidates[0]
                for path,h in [(item['source'],item['source_sha256']),(item['ranks_source'],item['ranks_sha256'])]:
                    assert sha(path)==h and cross[path]==h,(path,'different from cross input identity');identity[path]=h
                q=np.load(item['source'])[item['source_key']];ranks=np.load(item['ranks_source'])['ranks'];np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape))
                single,group=window_probabilities(q,ranks)
                # Independent origin-wise atom-index reconstruction of every cached probability.
                ss=[];gg=[]
                for qi in q:
                    safe=np.take_along_axis(qi.transpose(2,0,1),ranks//3,axis=0)<=10
                    ss.append(np.concatenate([safe[:,off:off+6].all(1).mean(0) for off in OFFSETS]))
                    gg.append([safe[:,off:off+6,g].all((1,2)).mean() for off in OFFSETS for g in GROUPS])
                np.testing.assert_array_equal(single,np.asarray(ss));np.testing.assert_array_equal(group,np.asarray(gg))
                if method=='fixed_B':
                    rawfile=ART/'seasonal_probability_v1'/f/f'predictions_gated_graph_atoms_{seed}_budget40.npz';paramfile=ART/'review_followup_v1/baselines'/f/f'calibration_context1_graph_{seed}.npz';params=np.load(paramfile);rawq=np.load(rawfile)['test'];old=apply_groups(rawq,params['legacy_frozen32'],float(fold['scale']),True)
                    np.testing.assert_array_equal(q,old)
                    for kind,v in [('single',single),('group',group)]:np.testing.assert_array_equal(v,np.load(ART/'review_followup_v1/baselines'/f/f'probabilities_context1_graph_{seed}_legacy_frozen32_natural.npz')[kind])
                    context_checks.append(dict(fold=f,seed=seed,context1_graph_legacy_equals_cross_fixed_B=True,maximum_atom_difference=0.,raw_source=str(rawfile),refit64_is_same_weights_but_new_calibration=True))
                np.savez_compressed(folder/f'probabilities_{method}_{seed}.npz',single=single,group=group)
                for kind,v in [('single',single),('group',group)]:prob[method][kind].append(v)
                identities.append(dict(label=method,cross_label='B' if method=='fixed_B' else 'W28',**item,cross_source_identity_match=True))
        for kind in ['single','group']:
            keys,ix=physical_mapping(ids,kind);pop[kind+'_keys']=keys;pop[kind+'_inverse']=ix
            for method in METHODS:prob[method][kind]=np.stack(prob[method][kind])
        anchor=times[pop['single_keys'][:,0].min()].normalize();k=int((times[pop['single_keys'][:,0].max()]-anchor).total_seconds()//(7*86400))+1;draw=rng.integers(0,k,(2000,k));np.savez_compressed(folder/'population.npz',**pop);np.savez_compressed(folder/'draws.npz',draws=draw,anchor=str(anchor));data[f]=(pop,prob,anchor,draw)
    absolute=[];decisions=[];paired=[]
    for scope in ['ALL',*FOLDS]:
        fs=FOLDS if scope=='ALL' else [scope]
        for kind in ['single','group']:
            for method in METHODS:
                lo=hi=knownloss=0.;n=nknown=0;counts={}
                for f in fs:
                    pop,p,anchor,draw=data[f];y=pop[kind+'_labels'];pa=p[method][kind];ix=pop[kind+'_inverse'];low,high,ct=event_losses(np.mean(pa**2,0),np.mean((pa-1)**2,0),y,ix);lo+=low.sum();hi+=high.sum();n+=int(ct.sum());known=np.isfinite(y);knownloss+=np.mean((pa-y)**2,0)[known].sum();nknown+=int(known.sum())
                    c=decision_counts(pa.mean(0),y,.8)
                    for name in ['accepted','tp','fp','fn','tn','unknown_accepted','unknown_rejected']:counts[name]=counts.get(name,0)+c[name]
                absolute.append(dict(scope=scope,kind=kind,method=method,n=n,known_records=nknown,unknown_records=n-nknown,known_brier=knownloss/nknown,lower=lo/n,upper=hi/n))
                decisions.append(dict(scope=scope,kind=kind,method=method,n=n,threshold=.8,**counts))
            for metric in ['brier','FP','FN']:
                pointlo=pointhi=0.;n=0;bl=np.zeros(2000);bu=np.zeros(2000);bn=np.zeros(2000);support=[]
                for f in fs:
                    pop,p,anchor,draw=data[f];y=pop[kind+'_labels'];ix=pop[kind+'_inverse'];a=p['rolling28'][kind];b=p['fixed_B'][kind]
                    if metric=='brier':zero=np.mean(a*a-b*b,0);one=np.mean((a-1)**2-(b-1)**2,0)
                    else:
                        da=(a.mean(0)>=.8-1e-12).astype(float);db=(b.mean(0)>=.8-1e-12).astype(float)
                        zero=da-db if metric=='FP' else np.zeros_like(da);one=np.zeros_like(da) if metric=='FP' else db-da
                    lo,hi,ct=event_losses(zero,one,y,ix);block=((times[pop[kind+'_keys'][:,0]]-anchor).total_seconds()//(7*86400)).astype(int);k=draw.shape[1];ls=np.bincount(block,weights=lo,minlength=k);us=np.bincount(block,weights=hi,minlength=k);ns=np.bincount(block,weights=ct,minlength=k)
                    pointlo+=lo.sum();pointhi+=hi.sum();n+=int(ct.sum());bl+=ls[draw].sum(1);bu+=us[draw].sum(1);bn+=ns[draw].sum(1);support.append(int((ns>0).sum()))
                good=bn>0;supported=min(support)>=4 and good.sum()>=1900
                paired.append(dict(scope=scope,kind=kind,candidate='rolling28',baseline='fixed_B',metric=metric,n=n,lower=pointlo/n,upper=pointhi/n,outer_low=float(np.quantile(bl[good]/bn[good],.025)) if supported else None,outer_high=float(np.quantile(bu[good]/bn[good],.975)) if supported else None,minimum_fold_blocks=min(support),valid_draws=int(good.sum()),supported=bool(supported),primary=scope=='ALL' and kind=='single' and metric=='brier'))
    for name,rows in [('absolute_brier.csv',absolute),('decision_counts.csv',decisions),('paired.csv',paired)]:pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    merged=pd.DataFrame(absolute).merge(pd.DataFrame(decisions),on=['scope','kind','method','n'],validate='one_to_one').rename(columns={'known_brier':'brier_known','n':'records'})
    merged.to_csv(OUT/'absolute_scores.csv',index=False,encoding='utf-8-sig')
    pairframe=pd.DataFrame(paired).rename(columns={'n':'records'})
    lookup={(r['scope'],r['kind'],r['method']):r['known_brier'] for r in absolute}
    pairframe['known_difference']=[lookup[r.scope,r.kind,'rolling28']-lookup[r.scope,r.kind,'fixed_B'] if r.metric=='brier' else np.nan for r in pairframe.itertuples()]
    pairframe[pairframe.metric=='brier'].to_csv(OUT/'paired_brier.csv',index=False,encoding='utf-8-sig')
    js(OUT/'version_identity.json',dict(exact_cross_methods=cp['methods'],sources=identities,context1_graph_identity_checks=context_checks,warning='precision/state_gate B and W28 are a different pair; never join labels across studies'))
    js(OUT/'source_identity.json',identity)
    js(OUT/'complete.json',dict(status='COMPLETE',unique_origins=880,probability_streams=24,source_and_rank_reconstructions=24,cross_source_matches=24,context1_graph_legacy_exact_matches=12,absolute_rows=len(absolute),decision_rows=len(decisions),paired_rows=len(paired)))
    assert sum(r['primary'] for r in paired)==1
    lines=['# 固定月校准与滚动affine校准：精确版本身份及结果','', '本表fixed_B精确映射seasonal_rolling_v1/inventory.json的model=gated_graph_atoms、method=frozen，source为seasonal_probability_v1各折calibrated_gated_graph_atoms_{seed}.npz的q；rolling28精确映射同一inventory的method=rolling28，其source为seasonal_rolling_v1各折predictions_gated_graph_atoms_{seed}_rolling28.npz的q。两者沿用相同冻结网络权重、原始float32年代校准结果与partial05秩，不重新拟合。评价单独使用原始float64真值。', '', '这对来源与cross_execution的B=frozen、W28=rolling28完全一致；24条源/秩SHA均对cross的source_identity严格匹配，并从原子/秩双路径重建全部窗口概率。它不是precision/state表的B=rolling28对W28=residual_warm WT。baselines/context1_graph的legacy_frozen32自然输入原子及窗口概率与这里fixed_B逐元素相同（12条），refit64沿用相同原始网络但校准参数不同。', '', '唯一主比较为ALL单机Brier，方向rolling28−fixed_B。四折880起报、3种子、四个6小时偏移；损失先逐种子计算再平均，0.8决策使用种子平均概率。物理窗口共享未知真值；按目标起始时刻7日块折内2000次（seed20261106），各折9块，保留末端部分块。仍是同年开发证据，外包不是选择校正后的正式总体置信保证。', '', '| ALL指标 | 有限样本缺测差界 | 目标时刻分块外包 |','|---|---|---|']
    for r in paired:
        if r['scope']=='ALL':lines.append(f"| {r['kind']} {r['metric']} | [{r['lower']:+.8f}, {r['upper']:+.8f}] | [{r['outer_low']:+.8f}, {r['outer_high']:+.8f}] |")
    lines+=['', '滚动affine相对固定月校准改善这对既有预测的Brier并大幅降低接受量/已知误判；单机已知漏窗同时增加。不能将这对结果替换为残差WT相对滚动affine的结论，也不能据此证明所有滚动校准普遍优越。绝对已知Brier与FP/FN见absolute_scores.csv，配对Brier见paired_brier.csv，全部Brier/FP/FN见paired.csv。']
    (OUT/'identity_and_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(paired).query("scope=='ALL'").to_string(index=False))
if __name__=='__main__':run()
