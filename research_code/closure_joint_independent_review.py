"""Read-only H2 audit: direct squared losses, physical keys, permutations."""
from pathlib import Path
from functools import lru_cache
from itertools import product
import hashlib,json,time
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
from lead_calibration_core import apply_groups
from closure_outage_independent_verify import labels_keys

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';SRC=ART/'review_closure_v1/joint_structure'
OUT=ART/'review_closure_v1/independent_review';OLD=ART/'review_followup_v1/baselines'
FOLDS=['M05','M07','M09','M11'];MODELS=['context1_graph','analog112'];SEEDS=[20241101,20241102,20241103]
OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]];SHUFFLES=[2026110201,2026110202,2026110203]
REPORT=ROOT.parent/'研究笔记/阶段收尾_2026-11-02/联合结构独立审查.md'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def direct_prob(q,ranks,threshold):
    # Independent original-atom lookup, without repeat/transposition/materialize.
    values=q[np.arange(72)[None,:,None],np.arange(17)[None,None,:],ranks//3]
    good=values<=threshold
    single=np.concatenate([good[:,o:o+6].all(1).mean(0) for o in OFFSETS])
    group=np.array([good[:,o:o+6,nodes].all((1,2)).mean() for o in OFFSETS for nodes in GROUPS])
    return single,group,values

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);tracked={str(Path(__file__)):sha(__file__)}
    for path,digest in read(SRC/'final_manifest.json').items():assert sha(path)==digest,path;tracked[path]=digest
    for path,digest in read(SRC/'source_identity.json').items():assert sha(path)==digest,path;tracked[path]=digest
    spec=read(SRC/'protocol.json');assert spec['permutation'].endswith('finite random pairing approximates independence')
    raw=load_truth()[2];assert raw.dtype==np.float64
    pop={f:dict(np.load(SRC/f'population_{f}.npz')) for f in FOLDS}
    cache={(f,m,s):dict(np.load(SRC/f'probabilities_{f}_{m}_{s}.npz')) for f,m,s in product(FOLDS,MODELS,SEEDS)}
    original_labels={}
    for f,kind,t in product(FOLDS,['single','group'],[8,10,12]):
        y,keys=labels_keys(raw,pop[f]['ids'],t,kind);np.testing.assert_array_equal(y,pop[f][f'{kind}_{t}']);original_labels[f,kind,t]=(y,keys)
    @lru_cache(None)
    def design(f,kind,t,offset):
        y,keys=original_labels[f,kind,t];width=17 if kind=='single' else 3
        cols=np.arange(4*width) if offset=='all' else np.arange(OFFSETS.index(int(offset))*width,(OFFSETS.index(int(offset))+1)*width)
        y,keys=y[:,cols],keys[:,cols]
        unique,inverse=np.unique(keys.reshape(-1,2),axis=0,return_inverse=True)
        truth=np.full(len(unique),np.nan);known=np.isfinite(y.ravel());truth[inverse[known]]=y.ravel()[known]
        np.testing.assert_array_equal(truth[inverse],y.ravel())
        return cols,inverse,truth,y
    @lru_cache(None)
    def primitive(f,kind,t,offset,model,seed,variant,rep):
        cols,ix,truth,y=design(f,kind,t,offset)
        p=cache[f,model,seed][f'{kind}_{t}_{variant}_{rep}'][:,cols]
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        # Every seed and shuffle is squared BEFORE averaging.
        z0=np.zeros(len(truth));z1=np.zeros(len(truth))
        np.add.at(z0,ix,(p*p).ravel());np.add.at(z1,ix,((p-1)*(p-1)).ravel())
        return z0,z1
    def evaluate(row,absolute=False):
        ff=FOLDS if row.scope=='ALL' else [row.scope];seeds=SEEDS if row.seed=='mean' else [int(row.seed)]
        numerator_lo=numerator_hi=knownsum=0.;records=knownrecords=0
        for f in ff:
            cols,ix,truth,y=design(f,row.kind,int(row.threshold),row.offset)
            if absolute:
                reps=[0] if row.variant=='original' else [1,2,3] if row.repeat=='mean' else [int(row.repeat)]
                samples=[primitive(f,row.kind,int(row.threshold),row.offset,row.model,s,row.variant,r) for s,r in product(seeds,reps)]
                zero=np.mean([v[0] for v in samples],axis=0);one=np.mean([v[1] for v in samples],axis=0)
            elif row.model=='graph-minus-analog':
                variant=row.comparison;reps=[0] if variant=='original' else [1,2,3]
                aa=[primitive(f,row.kind,int(row.threshold),row.offset,'context1_graph',s,variant,r) for s,r in product(seeds,reps)]
                bb=[primitive(f,row.kind,int(row.threshold),row.offset,'analog112',s,variant,r) for s,r in product(seeds,reps)]
                zero=np.mean([a[0]-b[0] for a,b in zip(aa,bb)],axis=0);one=np.mean([a[1]-b[1] for a,b in zip(aa,bb)],axis=0)
            else:
                variant=row.comparison.removeprefix('original-minus-');reps=[1,2,3] if row.repeat=='mean' else [int(row.repeat)]
                aa=[primitive(f,row.kind,int(row.threshold),row.offset,row.model,s,'original',0) for s,r in product(seeds,reps)]
                bb=[primitive(f,row.kind,int(row.threshold),row.offset,row.model,s,variant,r) for s,r in product(seeds,reps)]
                zero=np.mean([a[0]-b[0] for a,b in zip(aa,bb)],axis=0);one=np.mean([a[1]-b[1] for a,b in zip(aa,bb)],axis=0)
            known=np.isfinite(truth);fixed=np.where(truth==1,one,zero)
            numerator_lo+=fixed[known].sum()+np.minimum(zero[~known],one[~known]).sum()
            numerator_hi+=fixed[known].sum()+np.maximum(zero[~known],one[~known]).sum()
            knownsum+=fixed[known].sum();knownrecords+=int(np.isfinite(y).sum());records+=y.size
        return numerator_lo/records,numerator_hi/records,knownsum/knownrecords
    paired=pd.read_csv(SRC/'paired_brier.csv',dtype={'seed':str,'repeat':str,'offset':str})
    absolute=pd.read_csv(SRC/'absolute_brier.csv',dtype={'seed':str,'repeat':str,'offset':str})
    summary=pd.read_csv(SRC/'summary_effects.csv',dtype={'seed':str,'repeat':str,'offset':str})
    support=pd.read_csv(SRC/'label_support.csv',dtype={'offset':str})
    assert int(paired.primary.sum())==1
    maxerror=0.;counts={}
    for name,table,isabsolute in [('paired',paired,False),('absolute',absolute,True),('summary',summary,False)]:
        for row in table.itertuples():
            actual=evaluate(row,isabsolute)
            error=max(abs(actual[0]-row.lower),abs(actual[1]-row.upper),abs(actual[2]-row.known_brier_or_delta))
            assert error<1e-12,(name,row.Index,error);maxerror=max(maxerror,error)
        counts[name]=len(table);print('RECOMPUTED',name,len(table),round(time.monotonic()-start,1),flush=True)
    for row in support.itertuples():
        ff=FOLDS if row.scope=='ALL' else [row.scope]
        parts=[design(f,row.kind,int(row.threshold),row.offset) for f in ff]
        records=sum(p[3].size for p in parts);known=sum(np.isfinite(p[3]).sum() for p in parts)
        assert records==row.records and known==row.known and records-known==row.unknown
        assert sum(len(p[2]) for p in parts)==row.physical_windows
        assert sum(np.isnan(p[2]).sum() for p in parts)==row.unknown_physical_windows
        assert sum(len(p[3]) for p in parts)==row.origins
        assert abs((records-known)/records-row.unknown_rate)<1e-12
    # Independently regenerate the nine saved permutation indices.
    indices={}
    for kind,rep in product(['time','space','both'],[1,2,3]):
        ix=np.load(SRC/f'indices_{kind}_{rep}.npz')['indices'];rng=np.random.default_rng(SHUFFLES[rep-1])
        if kind=='time':expected=np.broadcast_to(np.array([rng.permutation(57) for _ in range(72)]).T[:,:,None],(57,72,17))
        elif kind=='space':expected=np.broadcast_to(np.array([rng.permutation(57) for _ in range(17)]).T[:,None,:],(57,72,17))
        else:expected=np.array([rng.permutation(57) for _ in range(72*17)]).T.reshape(57,72,17)
        np.testing.assert_array_equal(ix,expected)
        np.testing.assert_array_equal(np.sort(ix,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ix.shape))
        indices[kind,rep]=ix
    inv=read(OLD/'inventory.json');selected=[r for r in inv if r['family'] in MODELS and r['regime']=='natural' and r['method']=='refit64']
    assert len(selected)==24
    probability_checks=global_checks=value_checks=rank_checks=0
    for r in selected:
        f,m,s=r['fold'],r['family'],r['seed'];source=np.load(r['source'])[r['source_key']]
        q=apply_groups(source[[0,-1]],np.load(r['parameters'])['refit64'],float(np.load(ART/'seasonal_probability_v1'/f/'data.npz')['scale']),True)
        ranks=np.load(r['ranks'])['ranks'];assert ranks.shape==(57,72,17)
        np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape))
        variants=[('original',0,ranks)]
        for (kind,rep),ix in indices.items():
            rr=ranks[ix,np.arange(72)[None,:,None],np.arange(17)[None,None,:]]
            np.testing.assert_array_equal(np.sort(rr,axis=0),np.sort(ranks,axis=0));rank_checks+=1
            variants.append((kind,rep,rr))
        global_order=np.random.default_rng(SHUFFLES[0]).permutation(57)
        for j,pos in enumerate([0,len(source)-1]):
            original_values=direct_prob(q[j],ranks,10)[2]
            for kind,rep,rr in variants:
                for threshold in [8,10,12]:
                    sp,gp,values=direct_prob(q[j],rr,threshold)
                    np.testing.assert_array_equal(sp,cache[f,m,s][f'single_{threshold}_{kind}_{rep}'][pos])
                    np.testing.assert_array_equal(gp,cache[f,m,s][f'group_{threshold}_{kind}_{rep}'][pos]);probability_checks+=1
                np.testing.assert_array_equal(np.sort(values,axis=0),np.sort(original_values,axis=0));value_checks+=1
            for threshold in [8,10,12]:
                sp,gp,_=direct_prob(q[j],ranks[global_order],threshold)
                np.testing.assert_array_equal(sp,cache[f,m,s][f'single_{threshold}_original_0'][pos])
                np.testing.assert_array_equal(gp,cache[f,m,s][f'group_{threshold}_original_0'][pos]);global_checks+=1
    primary=paired[paired.primary].iloc[0]
    assert abs(primary.lower+0.06979954136443463)<1e-12 and abs(primary.upper+0.03630052905350281)<1e-12
    sec=summary[(summary.model=='context1_graph')&(summary.threshold==10)&(summary.kind=='group')&(summary.comparison=='original-minus-space')].iloc[0]
    note=ROOT.parent/'研究笔记/阶段收尾_2026-11-02/联合结构与阈值实验.md';text=note.read_text(encoding='utf-8')
    assert '有限情景的独立性近似' in text and '不能称为精确独立分布' in text
    assert 'M09原联合评分较差' in text and 'Graph仍差于强基线' in text
    reverse=paired[(paired.scope!='ALL')&(paired.seed=='mean')&(paired.repeat=='mean')&(paired.offset=='all')&(paired.model=='context1_graph')&(paired.comparison=='original-minus-time')&(paired.kind=='single')]
    reversal_rows=reverse[['scope','threshold','lower','upper']].to_dict('records')
    for path,digest in tracked.items():assert sha(path)==digest,path
    result=dict(status='PASS',paired_rows=counts['paired'],absolute_rows=counts['absolute'],summary_rows=counts['summary'],support_rows=len(support),
                max_score_recompute_error=maxerror,unique_primary=int(paired.primary.sum()),
                primary=dict(lower=float(primary.lower),upper=float(primary.upper)),secondary=dict(lower=float(sec.lower),upper=float(sec.upper)),
                probability_direct_atom_lookup_checks=probability_checks,global_negative_control_checks=global_checks,
                actual_value_multiset_checks=value_checks,full_coordinate_marginal_rank_checks=rank_checks,permutation_indices_reproduced=9,
                square_before_seed_and_shuffle_average=True,all_raw64_labels_recomputed=True,
                month_threshold_main_effects=reversal_rows,report_semantic_checks='PASS: finite approximation; M09 reversal and graph-minus-analog weakness disclosed; development-only conservative event-label bounds',
                scope='Different physical windows/thresholds/event types not constrained together. Same-event labels shared, finite scenario permutations approximate independence; no confidence or deployment claim.',
                source_hashes_read_only=tracked,elapsed_seconds=time.monotonic()-start)
    (OUT/'joint.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 联合结构独立审查','','独立复核通过：全部13740行配对评分、12720行绝对评分、60行summary与150行标签数。直接将每个保存seed及shuffle概率平方，再平均平方损失；按绝对(目标窗口起始小时,节点/节点组)共享标签，对未知目标分别枚举0/1费用端点汇总。没有导入coefficients或sharp_bounds，没有先平均概率再平方。最大重算误差 '+format(maxerror,'.3g')+'。','',f'唯一主要登记为Graph阈值10、single6h、原联合−time、ALL880：[{primary.lower:.9f}, {primary.upper:.9f}]。共同6h原联合−space辅助界：[{sec.lower:.9f}, {sec.upper:.9f}]。','',f'9个保存索引由登记随机种子独立再生；216次全坐标秩多重集、480次实际值多重集一致。24来源×首末起报×十变体×三阈值共{probability_checks}组事件概率由原子索引直接重算精确一致；全局整轨迹重排{global_checks}组负控确实重新执行且精确不变。','',
      '全部边际57整数索引在每lead,node为完整排列，因此对任意起报的原子值保持多重集及边际CRPS。time保留同期跨节点向量，space保留单节点全轨迹；57条有限情景随机配对只近似消除相关，不能称精确独立分布。','','主文稿已明确披露M09反向及Graph差于analog112；不是挑出改善时间段。fold为双月评价段：M05=5–6月[2024-05-01,2024-07-01)，M07=7–8月[2024-07-01,2024-09-01)，M09=9–10月[2024-09-01,2024-11-01)，M11=11–12月[2024-11-01,2025-01-01)，均左闭右开。M09反向不能称为9月单月反向。主比较按fold与阈值如下：','','| fold(双月) | 阈值 | 下界 | 上界 |','|---|---:|---:|---:|']
    lines += [f"| {r['scope']} | {r['threshold']:g} | {r['lower']:.9f} | {r['upper']:.9f} |" for r in reversal_rows]
    lines += ['','ALL阈值8/10/12的主比较界均为负仅支持开发集平均改善，不代表每个双月段或每任务普遍改善。Graph−analog采用相同Graph秩模板，仍反映冻结边际质量差，不能独立归因为图结构。','',
      '界仅对同一完整物理窗口共标签；没有跨窗口共享小时AND、跨阈值嵌套、单机/组窗口包含关系，因此是保守识别范围，不是完整物理约束下sharp界、置信区间、独立外推或部署保证。旧输入与全部原产物SHA重读一致且只读。','',
      '复现：在风电运维Demo目录运行 `python closure_joint_independent_review.py`。核验结果及来源摘要在artifacts/review_closure_v1/independent_review/joint.json。']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes_read_only','month_threshold_main_effects']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
