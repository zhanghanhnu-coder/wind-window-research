"""Independent read-only endpoint and weighting review. No original evaluators imported."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,argparse
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ART=BASE/'artifacts'
SRC=ART/'review_closure_v1/update_robustness'
OUT=ART/'review_closure_v1/independent_review'
TRUTH=BASE.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl'
MATCHED=ART/'review_iteration3_v1/matched_updates/matched_windows.csv'
BAD=ART/'review_iteration2_v1/data_gaps/partial_hour_evidence.npz'
NOTE=BASE.parent/'研究笔记/阶段收尾_2026-11-02/更新诊断独立审查.md'
SIGN={'interaction':[-1,1,1,-1],'initial_model_difference':[1,-1,0,0],'current_model_difference':[0,0,1,-1],'analog_update':[-1,0,1,0],'graph_update':[0,-1,0,1]}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def weights(rows,estimand):
    """Build explicit nested allocation buckets instead of using original group transforms."""
    outer=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for pos,r in enumerate(rows.itertuples()):
        origin=(r.fold,int(r.issue));event=(int(r.issue+r.start+1),int(r.node),int(r.duration))
        if estimand=='equal_origin':outer[origin]['single'][int(r.seed)].append(pos)
        else:outer[event][origin][int(r.seed)].append(pos)
    w=np.zeros(len(rows))
    for opportunities in outer.values():
        for seeds in opportunities.values():
            for indices in seeds.values():w[indices]=1/(len(outer)*len(opportunities)*len(seeds)*len(indices))
    assert abs(w.sum()-1)<1e-12
    for opportunities in outer.values():
        positions=[i for seeds in opportunities.values() for bucket in seeds.values() for i in bucket]
        assert abs(w[positions].sum()-1/len(outer))<1e-12
    return w

def main(full=False):
    OUT.mkdir(parents=True,exist_ok=True)
    source_files=[BASE/'closure_robustness.py',BASE/'iteration2_bounds_core.py',BASE/'iteration3_matched_core.py',MATCHED,TRUTH,BAD,Path(__file__),SRC/'protocol.json']
    hashes={str(p):sha(p) for p in source_files}
    d=pd.read_csv(MATCHED);raw=pd.read_pickle(TRUTH)['max'].to_numpy(copy=True);bad=np.load(BAD)['known_bad'].astype(bool)
    assert raw.dtype==np.float64 and raw.shape==bad.shape and len(d)==9150
    fix=~np.isfinite(raw)&bad;assert fix.sum()==494
    partial=raw.copy();partial[fix]=11.
    np.testing.assert_array_equal(partial[np.isfinite(raw)],raw[np.isfinite(raw)])
    # Independent raw labels and target index construction, never window_term.
    def observations(weather,rows):
        dur=rows.duration.to_numpy(int);starts=rows.issue.to_numpy(int)+rows.start.to_numpy(int)+1
        hours=starts[:,None]+np.arange(dur.max())[None,:]
        values=weather[hours,rows.node.to_numpy(int)[:,None]]
        validpos=np.arange(dur.max())[None,:]<dur[:,None]
        return values,validpos
    values,active=observations(raw,d)
    fail=((values>10)&active).any(1);known=(np.isfinite(values)|~active).all(1)
    labels=np.where(fail,0.,np.where(known,1.,np.nan));np.testing.assert_array_equal(labels,d.label.to_numpy())
    pv,_=observations(partial,d);pfail=((pv>10)&active).any(1);pknown=(np.isfinite(pv)|~active).all(1)
    plabels=np.where(pfail,0.,np.where(pknown,1.,np.nan));changed=np.isnan(labels)&np.isfinite(plabels)
    assert changed.sum()==3 and (plabels[changed]==0).all()
    scopes=['ALL']+['leave_fold_'+f for f in ['M05','M07','M09','M11']]
    endpoints=[];maxerror=0.;direct_certs=[]
    for scope in scopes:
        sub=d if scope=='ALL' else d[d.fold!=scope.removeprefix('leave_fold_')]
        for estimand in ('equal_origin','physical_event'):
            w=weights(sub,estimand);p=sub[['AI','GI','AC','GC']].to_numpy()
            for evidence,weather in [('raw64',raw),('partial_bad_only',partial)]:
                for comparison,sign in SIGN.items():
                    path=SRC/'certificates'/f'{scope}__{estimand}__{evidence}__{comparison}.json'
                    cert=read(path);hashes[str(path)]=sha(path);direct_certs.append(cert)
                    for side in ('lower','upper'):
                        states=cert['optimization'][side+'_witness'];completed=weather.copy();seen=set()
                        for h,n,state in states:
                            assert state in [0,1] and (h,n) not in seen and not np.isfinite(weather[h,n]);seen.add((h,n))
                            completed[h,n]=0. if state else 11.
                        completed[~np.isfinite(completed)]=11.
                        obs,active=observations(completed,sub);y=((obs<=10)|~active).all(1).astype(float)
                        four_squares=np.square(p-y[:,None])
                        actual=float(sum(w[i]*sum(float(four_squares[i,j])*sign[j] for j in range(4)) for i in range(len(w))))
                        err=abs(actual-cert[side]);assert err<1e-10;maxerror=max(maxerror,err)
                        endpoints.append(dict(scope=scope,estimand=estimand,evidence=evidence,comparison=comparison,side=side,stored=cert[side],independent=actual,error=err))
    old=pd.read_csv(ART/'review_iteration3_v1/matched_updates/bounds_summary.csv')
    old=old[(old.population=='all44_primary')&(old.scope=='ALL')&(old.start_group=='ALL')]
    for comparison in SIGN:
        r=old[old.comparison==comparison].iloc[0]
        cert=next(c for c in direct_certs if c['scope']=='ALL' and c['estimand']=='equal_origin' and c['evidence']=='raw64' and c['comparison']==comparison)
        assert abs(r.shared_lower-cert['lower'])<1e-12 and abs(r.shared_upper-cert['upper'])<1e-12
    lm09=next(c for c in direct_certs if c['scope']=='leave_fold_M09' and c['estimand']=='equal_origin' and c['evidence']=='raw64' and c['comparison']=='interaction')
    assert lm09['lower']<0<lm09['upper']
    allfiles=list((SRC/'certificates').glob('*.json'));statuschecks=0;nested=0;summary={};full_complete=False
    if full:
        assert (SRC/'verification.json').exists() and len(allfiles)==1120,'full artifacts not complete'
        allcert={}
        for path in allfiles:
            c=read(path);hashes[str(path)]=sha(path);assert c['certificate']=='PASS' and c['lower']<=c['upper']+1e-10
            for side in ['lower','upper']:
                cc=c['optimization'][side+'_certificate']
                assert cc['status']==0 and cc['absolute_gap']<1e-7 and cc['mip_gap']<1e-9
                for key in ['primal','dual_bound','witness_recomputed','direct_four_square_weather_value']:assert abs(cc[key]-c[side])<1e-7
                statuschecks+=1
            allcert[c['scope'],c['estimand'],c['evidence'],c['comparison']]=c
        for key,a in allcert.items():
            if key[2]!='raw64':continue
            b=allcert[key[0],key[1],'partial_bad_only',key[3]]
            assert a['lower']-1e-9<=b['lower']<=b['upper']+1e-9 and b['upper']<=a['upper']+1e-9;nested+=1
        df=pd.read_csv(SRC/'bounds_summary.csv');assert len(df)==1120
        for r in df.itertuples():
            c=allcert[r.scope,r.estimand,r.evidence,r.comparison];assert abs(r.lower-c['lower'])<1e-14 and abs(r.upper-c['upper'])<1e-14
        for evidence in ['raw64','partial_bad_only']:
            for estimand in ['equal_origin','physical_event']:
                q=df[(df.comparison=='interaction')&(df.evidence==evidence)&(df.estimand==estimand)]
                seeds=q[q.kind=='seed'];loo=q[q.kind=='leave_issue']
                summary[evidence+'__'+estimand]=dict(seeds=seeds[['scope','lower','upper']].to_dict('records'),leave_issue=dict(n=len(loo),lower_min=float(loo.lower.min()),lower_max=float(loo.lower.max()),upper_min=float(loo.upper.min()),upper_max=float(loo.upper.max()),cross_zero=int(((loo.lower<=0)&(loo.upper>=0)).sum())))
        note=BASE.parent/'研究笔记/阶段收尾_2026-11-02/更新诊断稳健性.md'
        text=note.read_text(encoding='utf-8');hashes[str(note)]=sha(note)
        eq=summary['raw64__equal_origin']['leave_issue']
        for value in [lm09['lower'],lm09['upper'],eq['lower_min'],eq['lower_max'],eq['upper_min'],eq['upper_max']]:assert f'{value:.8f}' in text
        for seed in summary['raw64__equal_origin']['seeds']:
            line=next(line for line in text.splitlines() if line.strip().startswith('| '+seed['scope']))
            parts=[x.strip() for x in line.strip().strip('|').split('|')]
            assert abs(float(parts[-2])-seed['lower'])<1e-7 and abs(float(parts[-1])-seed['upper'])<1e-7
        for key,c in allcert.items():
            if key[0]=='ALL' and key[3]=='interaction':
                options=[[x.strip() for x in line.strip().strip('|').split('|')] for line in text.splitlines() if line.startswith('|')]
                row=next(parts for parts in options if parts[:2]==[key[1],key[2]])
                assert abs(float(row[-2])-c['lower'])<1e-7 and abs(float(row[-1])-c['upper'])<1e-7
        assert '跨0' in text and '不能解释为置信区间' in text and '不同estimand' in text
        full_complete=True
    for p,h in hashes.items():assert sha(p)==h,p
    result=dict(status='PASS' if full_complete else 'PARTIAL_PASS_FULL_PENDING',source_hashes=hashes,independent_certificates=100,independent_endpoints=len(endpoints),max_endpoint_error=maxerror,old_all_endpoints_reproduced=10,raw_label_rows_checked=len(d),unknown_bad_fixed=int(fix.sum()),changed_rows=int(changed.sum()),all_certificates_visible=len(allfiles),solver_certificate_endpoint_status_checks=statuschecks,nesting_checks=nested,full_complete=full_complete,leave_M09_raw_equal_origin={k:lm09[k] for k in ['lower','upper']},influence_summary=summary,limitations='No independent MILP rerun; 100 certificates have independently reconstructed weights/weather/four-square endpoints. Full run reads status/dual-gap and nesting for all1120, not independently solving optimality. Bounds are conditional identification ranges, not CI.')
    pd.DataFrame(endpoints).to_csv(OUT/'robustness_endpoints.csv',index=False,encoding='utf-8-sig')
    (OUT/'robustness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 更新诊断独立审查','',f"状态：{result['status']}。源与旧产物只读，审查代码不导入closure_robustness、evaluate_bounds、aggregation_weights、window_term或solve_bounds。",'',
    '独立以matched_windows.csv四格概率、原始float64小时资料及known_bad重建标签与天气。自行按显式嵌套桶分配两种权重：等起报→种子→窗口；物理事件等权→预测机会→种子→重复task行。四格逐行先平方，再应用五组对比系数和权重。物理事件权重改变目标量，不能解释为对原等起报估计的无偏修正。',
    '',f"ALL及四leave-fold × 两证据 × 两权重 × 五对比，共100证书、200端点独立重算，最大误差{maxerror:.3g}；原ALL五对比十端点复现。删除M09后原等起报交互[{lm09['lower']:.8f}, {lm09['upper']:.8f}]跨0。",'',
    'partial仅将494个raw64未知且known_bad小时固定失败，所有原已知值保持不变；9150行raw64标签独立复核，3行未知窗口转为确定失败。可行集由原集合与固定失败约束相交，故嵌套。原求解器AND约束逐项阅读，窗口变量≤每小时状态且≥各状态和−小时数+1；状态均二元，故正确表达共享小时合取。', '',
    f'完整证书读取：{len(allfiles)}；端点status/dual-gap核对{statuschecks}项；raw/partial嵌套核对{nested}项。'+('1120份已齐，全量CSV与证书一致；三个seed与44留起报的全部四种证据/权重组合范围见robustness.json。稿件中删除M09界、LOO范围与限制表述已逐项核对。' if full_complete else '完整计算仍在进行，当前没有宣称全部1120份已审；完成后须以--full重跑。'), '',
    'fold定义来自seasonal_probability_core.py：M05=2024年5–6月，M07=7–8月，M09=9–10月，M11=11–12月；均为双月评价段，M09不等同于9月单月。', '',
    '本审查没有重跑MILP；独立端点重算证明给定共同天气见证及权重的评分一致，最优性依赖现有求解器证书与已审查的AND模型。种子属于同一天气的秩敏感性，留起报/fold是开发影响诊断；范围不是置信区间。', '',
    '完整源SHA与逐端点误差：artifacts/review_closure_v1/independent_review/robustness.json及robustness_endpoints.csv。复现：python 风电运维Demo/closure_robustness_independent_review.py --full。', '',
    '被审主源SHA256：'+hashes[str(BASE/'closure_robustness.py')], '审查脚本SHA256：'+hashes[str(Path(__file__))]]
    if full_complete:
        eq=summary['raw64__equal_origin'];loo=eq['leave_issue']
        lines+=['','## 交互的种子与留起报范围',
        'raw64等起报三个seed界分别为：'+ '；'.join(f"{r['scope']} [{r['lower']:.8f}, {r['upper']:.8f}]" for r in eq['seeds'])+'，均严格为正。',
        f"44留起报下界范围[{loo['lower_min']:.8f}, {loo['lower_max']:.8f}]，上界范围[{loo['upper_min']:.8f}, {loo['upper_max']:.8f}]，跨0共{loo['cross_zero']}例。删除单一起报不改变正号，删除M09双月fold跨0，扰动尺度不同。",
        'raw64/partial × 两权重的其余三套组合也均为三个seed严格正、44留起报均不跨0；数值全保存在JSON。不能将这些重复解释为独立天气复现或概率覆盖率。']
    NOTE.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('REVIEW',result['status'],'endpoints',len(endpoints),'maxerr',maxerror,'certificates',len(allfiles),'nested',nested)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args();main(a.full)
