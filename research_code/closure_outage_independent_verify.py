"""Independent outage/threshold table, raw-label, inference and zero-input audit."""
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
import pandas as pd
import torch
from precision_truth_core import load_truth
from crps_distribution_core import AtomNet
from lead_calibration_core import apply_groups

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts'
OUT=ART/'review_closure_v1/outage_stress';TH=ART/'review_closure_v1/threshold_h1';OLD=ART/'review_followup_v1/baselines'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
FAMILIES=['context1_graph','context1_mean','context1_own'];OFFSETS=[0,24,48,66]
GROUPS=[[0,1,2],[8,9,10],[13,15,16]]
MODELS=dict(context1_graph='gated_graph_atoms',context1_mean='gated_mean_atoms',context1_own='gated_temporal_atoms')
REPORT=ROOT.parent/'研究笔记/阶段收尾_2026-11-02/全场中断与阈值_独立审查.md'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def labels_keys(raw,ids,threshold,kind):
    labels=[];key_columns=[]
    for offset in OFFSETS:
        chunk=raw[ids[:,None]+np.arange(offset+1,offset+7)]
        selections=[[n] for n in range(17)] if kind=='single' else GROUPS
        for index,nodes in enumerate(selections):
            v=chunk[:,:,nodes]
            labels.append(np.where((np.isfinite(v)&(v>threshold)).any((1,2)),0.,np.where(np.isfinite(v).all((1,2)),1.,np.nan)))
            key_columns.append(np.column_stack([ids+offset+1,np.full(len(ids),index,dtype=int)]))
    return np.stack(labels,1),np.stack(key_columns,1)

def recompute(a,b,y,keys):
    # Sum the two possible directly squared losses by physical event. Neither
    # loss_event_totals nor closure_outage_core.paired_bounds is imported.
    flatkeys=keys.reshape(-1,2);unique,inverse=np.unique(flatkeys,axis=0,return_inverse=True)
    loss0=np.mean(a*a-b*b,axis=0).ravel()
    loss1=np.mean((a-1)**2-(b-1)**2,axis=0).ravel()
    totals0=np.zeros(len(unique));totals1=np.zeros(len(unique))
    np.add.at(totals0,inverse,loss0);np.add.at(totals1,inverse,loss1)
    truth=np.full(len(unique),np.nan);known=np.isfinite(y.ravel());truth[inverse[known]]=y.ravel()[known]
    np.testing.assert_array_equal(truth[inverse],y.ravel())
    lo=np.where(truth==0,totals0,np.where(truth==1,totals1,np.minimum(totals0,totals1))).sum()/y.size
    hi=np.where(truth==0,totals0,np.where(truth==1,totals1,np.maximum(totals0,totals1))).sum()/y.size
    return float(lo),float(hi),len(unique)

def scalar_probs(q,ranks,threshold):
    single=np.zeros((4,17),int);group=np.zeros((4,3),int)
    for scenario in range(57):
        for k,off in enumerate(OFFSETS):
            good=np.array([all(q[h,n,int(ranks[scenario,h,n])//3]<=threshold for h in range(off,off+6)) for n in range(17)])
            single[k]+=good
            group[k]+=[good[nodes].all() for nodes in GROUPS]
    return single.ravel()/57,group.ravel()/57

def main():
    torch.set_num_threads(2);raw=load_truth()[2];assert raw.dtype==np.float64
    tracked={str(Path(__file__)):hashfile(__file__)};source_checked=0
    for directory in [OUT,TH]:
        manifest=read(directory/'source_identity.json')
        for path,digest in manifest.items():
            assert hashfile(path)==digest,path;tracked[path]=digest;source_checked+=1
    population={};data={};labels={};prob={};zero=[];poison_checks=0
    for fold in FOLDS:
        data[fold]=dict(np.load(ART/'seasonal_probability_v1'/fold/'data.npz'))
        f=data[fold];ids=f['test_ids'];population[fold]=dict(ids=ids,complete=np.array([(f['x'][i-23:i+1,:,2]==1).all() for i in ids]))
        old=dict(np.load(OLD/fold/'population.npz'));np.testing.assert_array_equal(ids,old['ids'])
        for kind in ['single','group']:
            for threshold in [8,10,12]:
                labels[fold,kind,threshold]=labels_keys(raw,ids,threshold,kind)
            np.testing.assert_array_equal(labels[fold,kind,10][0],old[kind+'_labels'])
        # Test no future read independently, without using outage_history.
        for pos in [0,len(ids)//2,len(ids)-1]:
            issue=int(ids[pos]);clean=f['x'][issue-23:issue+1].copy()
            contaminated=f['x'].copy();contaminated[issue+1:]=np.nan
            for hours in [0,6,12,24]:
                a=clean.copy();b=contaminated[issue-23:issue+1].copy()
                if hours:a[-hours:]=0;b[-hours:]=0
                keep=(a[...,2]>0);a=np.where(keep[...,None],a,0)
                keep=(b[...,2]>0);b=np.where(keep[...,None],b,0)
                np.testing.assert_array_equal(a,b);poison_checks+=1
        for family,seed in product(FAMILIES,SEEDS):
            for hours in [0,6,12,24]:
                path=OUT/f'{fold}_{family}_{seed}_outage{hours}.npz';tracked[str(path)]=hashfile(path)
                pack=dict(np.load(path));np.testing.assert_array_equal(pack['ids'],ids)
                for kind in ['single','group']:
                    a=pack[kind];assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
                    prob[fold,family,seed,hours,kind]=a
                    if hours==0:np.testing.assert_array_equal(a,np.load(OLD/fold/f'probabilities_{family}_{seed}_refit64_natural.npz')[kind])
                    if hours==24:
                        delta=float(np.max(np.abs(a-a[:1])));assert delta==0
                        zero.append(dict(fold=fold,family=family,seed=seed,kind=kind,max_across_origin_difference=delta))
        for family,seed in product(FAMILIES[:2],SEEDS):
            path=TH/f'{fold}_{family}_{seed}.npz';tracked[str(path)]=hashfile(path);z=dict(np.load(path))
            for threshold,kind in product([8,10,12],['single','group']):
                prob[fold,family,seed,threshold,'threshold',kind]=z[f'{kind}_{threshold}']
    maxerr=0.;outage_rows=pd.read_csv(OUT/'paired_bounds.csv');threshold_rows=pd.read_csv(TH/'paired_bounds.csv')
    for kind_of_table,table in [('outage',outage_rows),('threshold',threshold_rows)]:
        for row in table.itertuples():
            ff=FOLDS if row.scope=='ALL' else [row.scope]
            masks={f:population[f]['complete'] if row.cohort=='complete_input392' else np.ones(len(population[f]['ids']),bool) for f in ff}
            threshold=10 if kind_of_table=='outage' else row.threshold
            y=np.concatenate([labels[f,row.kind,threshold][0][masks[f]] for f in ff])
            keys=np.concatenate([labels[f,row.kind,threshold][1][masks[f]] for f in ff])
            def load(family,h):
                return np.stack([np.concatenate([prob[f,family,seed,h,row.kind][masks[f]] if kind_of_table=='outage' else prob[f,family,seed,h,'threshold',row.kind][masks[f]] for f in ff]) for seed in SEEDS])
            if kind_of_table=='threshold':a,b=load('context1_graph',threshold),load('context1_mean',threshold)
            elif row.contrast=='graph_minus_mean':a,b=load('context1_graph',row.outage_hours),load('context1_mean',row.outage_hours)
            elif row.contrast=='graph_minus_own':a,b=load('context1_graph',row.outage_hours),load('context1_own',row.outage_hours)
            else:
                family=row.contrast.removesuffix('_outage_minus_natural');a,b=load(family,row.outage_hours),load(family,0)
            lo,hi,physical=recompute(a,b,y,keys)
            error=max(abs(lo-row.lower),abs(hi-row.upper));assert error<1e-12;maxerr=max(maxerr,error)
            assert physical==row.physical_windows
            assert np.isfinite(y).sum()==row.known_records and np.isnan(y).sum()==row.unknown_records
            assert len(y)==row.origins
            if kind_of_table=='outage':
                assert y.size==row.forecast_records
                known=np.isfinite(y)
                if known.any():assert abs(np.mean((a[:,known]-y[known])**2-(b[:,known]-y[known])**2)-row.known_only_difference)<1e-12
            else:assert y.size==row.records
    # Original shared-event endpoint algebra tested against explicit common-label enumeration.
    a=np.array([[[.9,.1,.7]],[[.6,.4,.2]]]);b=np.array([[[.1,.9,.4]],[[.2,.8,.5]]]);y=np.full((1,3),np.nan);keys=np.array([[[1,0],[1,0],[7,1]]])
    lo,hi,_=recompute(a,b,y,keys)
    brute=[]
    for first,second in product([0,1],repeat=2):
        yy=np.array([[first,first,second]]);brute.append(float(np.mean((a-yy)**2-(b-yy)**2)))
    assert abs(lo-min(brute))<1e-12 and abs(hi-max(brute))<1e-12
    # Independent direct model invocation for each outage length and graph/mean.
    inference=[];fold='M05';seed=SEEDS[0];f=data[fold];ids=f['test_ids'];positions=[0,len(ids)//2,len(ids)-1]
    for family,hours in product(FAMILIES[:2],[6,12,24]):
        path=ART/'seasonal_probability_v1'/fold/f'{MODELS[family]}_{seed}_budget40.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        net=AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True);net.load_state_dict(ck['state']);net.eval()
        hist=np.stack([f['x'][int(ids[i])-23:int(ids[i])+1] for i in positions]).copy()
        hist[:,-hours:]=0;hist=np.where((hist[...,2]>0)[...,None],hist,0).astype(np.float32)
        assert (hist[:,-hours:]==0).all()
        if hours==24:assert (hist==0).all()
        with torch.no_grad():qraw=net(torch.from_numpy(hist)).numpy()
        theta=np.load(OLD/fold/f'calibration_{family}_{seed}.npz')['refit64']
        q=apply_groups(qraw,theta,float(f['scale']),True)
        ranks=np.load(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz')['ranks']
        prob_error=0.
        for j,pos in enumerate(positions):
            single,group=scalar_probs(q[j],ranks,10)
            for kind,values in [('single',single),('group',group)]:
                error=float(np.max(np.abs(values-prob[fold,family,seed,hours,kind][pos])))
                assert error==0;prob_error=max(prob_error,error)
        inference.append(dict(fold=fold,seed=seed,family=family,outage_hours=hours,positions=positions,
                              natural_refit64=True,original_same_rank_source=True,probability_max_error=prob_error))
    assert len(outage_rows)==400 and len(threshold_rows)==60 and len(zero)==72
    assert sum(p['complete'].sum() for p in population.values())==392
    result=dict(status='PASS',outage_rows_recomputed=len(outage_rows),threshold_rows_recomputed=len(threshold_rows),
                independent_table_max_error=maxerr,source_hash_checks=source_checked,old_sources_read_only=True,
                zero_input_constant_checks=zero,inference_source_checks=inference,independent_causal_poison_checks=poison_checks,
                natural_zero_outage_exact_streams=36,shared_unknown_enumerated_completions=4,
                warning='At 24h all fine input channels are zero. Per fold/family/seed all forecasts are constant across origins. Graph-minus-mean difference reflects frozen model/calibration priors and must not be evidence of real-time spatial information.')
    write(OUT/'independent_verification.json',result)
    for path,digest in tracked.items():assert hashfile(path)==digest,path
    write(OUT/'independent_sources_sha256.json',tracked)
    REPORT.write_text('# 全场中断与阈值：独立审查\n\n全部400行中断配对表与60行阈值配对表由直接平方误差及绝对物理窗口键独立重算，未调用 paired_bounds 或 loss_event_totals。严格raw64标签、完整输入392起报与ALL880、记录数、物理窗口数及已知/未知数量均复核。最大表差 '+format(maxerr,'.3g')+'。\n\n24小时全场中断使全部24小时细输入（值与mask）全零。4fold×3模型×3种子×single/group共72组预测跨起报逐元素完全常量。此时graph−mean差来自冻结模型及冻结自然refit64校准的先验差，不能解释成实时空间信息证据，不能用于声称图模型从当下邻居传播恢复了信息。\n\n6/12/24小时各对graph和mean独立调用原checkpoint，以首/中/末三个起报抽查，保持原自然refit64与相同rank文件；标量逐情景、逐小时生成概率，与已存概率全部精确一致。自然0小时36流精确复现，独立未来毒化检查48次通过。共享未知小例穷举4种完成一致。\n\n阈值8/10/12保持原权重、自然refit64与rank；阈值变化是原单位代理敏感性，不是物理安全阈值认证、新校准或独立验证。全场中断是人为输入压力，不代表自然缺测机制或可行动资格；此处事件界按完整物理窗口共标签，没有额外跨不同窗口的逐小时AND约束。来源SHA重读相符，旧来源全程只读。\n\n复现：在风电运维Demo目录运行 `python closure_outage_independent_verify.py`。核验及独立来源摘要在outage_stress/independent_verification.json和independent_sources_sha256.json。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['zero_input_constant_checks','inference_source_checks']},ensure_ascii=False))

if __name__=='__main__':main()
