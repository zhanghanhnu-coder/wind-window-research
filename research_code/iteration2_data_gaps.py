"""Recover valid partial-hour evidence without filling unobserved wind safely."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,SOURCE

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OUT=ROOT/'artifacts/review_iteration2_v1/data_gaps'
DOC=BASE/'研究笔记/第二轮实证推进_2026-11-02'
SOURCE_DIR=BASE/'xx风电场 2024.01.01-2024.12.31 (2)'
FIELD='feature_A'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(field=FIELD,threshold=10.,cleaning='reject abs in [1e-45,1e-30], negative, >60; no rounding',
                  grid='2024 five-minute left-closed hours; right-end hourly labels; Asia/Shanghai',
                  interpretation='Partial valid observation above threshold forces failure only. Partial observations below threshold never establish safety.',
                  scope='Existing original CSV evidence recovered from strict aggregation, not new observation or independent validation.',
                  primary_truth_unchanged=True)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol
    else:p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    times,nodes,raw=load_truth();identity={str(SOURCE):sha(SOURCE)}
    parts={node:[] for node in nodes};seen_files=[]
    for path in sorted(SOURCE_DIR.rglob('*.csv')):
        if '资产' in path.name:continue
        d=pd.read_csv(path,usecols=['k_device','k_ts',FIELD]);dev=d.k_device.astype(str).str[-2:]
        assert dev.nunique()==1;node=dev.iloc[0]
        if node not in parts:continue
        identity[str(path)]=sha(path)
        number=pd.to_numeric(d.k_ts,errors='coerce');stamp=pd.to_datetime(number,unit='ms',utc=True).dt.tz_convert('Asia/Shanghai').copy()
        missing=number.isna()
        if missing.any():stamp.loc[missing]=pd.to_datetime(d.loc[missing,'k_ts']).dt.tz_localize('Asia/Shanghai')
        assert stamp.notna().all() and not stamp.duplicated().any()
        assert ((stamp.dt.minute%5==0)&(stamp.dt.second==0)&(stamp.dt.microsecond==0)).all()
        value=pd.to_numeric(d[FIELD],errors='coerce');invalid=value.abs().between(1e-45,1e-30)|(value<0)|(value>60)
        value=value.mask(invalid)
        parts[node].append(pd.DataFrame(dict(value=value.to_numpy(),source=str(path.relative_to(BASE))),index=pd.DatetimeIndex(stamp)))
        seen_files.append(dict(file=str(path.relative_to(BASE)),node=node,rows=len(d)))
    grid=pd.date_range('2024-01-01','2025-01-01',inclusive='left',freq='5min',tz='Asia/Shanghai')
    ends=pd.date_range('2024-01-01 01:00','2025-01-01',freq='h',tz='Asia/Shanghai')
    counts=np.zeros(raw.shape,dtype=np.int16);maximum=np.full(raw.shape,np.nan);bad_rows=[];source_by_hour={}
    for j,node in enumerate(nodes):
        frame=pd.concat(parts[node]).sort_index();assert not frame.index.duplicated().any()
        aligned=frame.reindex(grid);values=aligned.value.to_numpy().reshape(-1,12);good=np.isfinite(values)
        count=good.sum(1);pmax=np.where(good,values,-np.inf).max(1);pmax[count==0]=np.nan
        series_count=pd.Series(count,index=ends).reindex(times).fillna(0).astype(int)
        series_max=pd.Series(pmax,index=ends).reindex(times)
        counts[:,j]=series_count.to_numpy();maximum[:,j]=series_max.to_numpy()
        reconstructed=np.where(counts[:,j]==12,maximum[:,j],np.nan)
        np.testing.assert_allclose(reconstructed,raw[:,j],atol=1e-12,rtol=0,equal_nan=True)
        # Independent pandas aggregation checks partial counts/max, not only complete hours.
        other=frame.value.resample('1h',closed='left',label='right').agg(['count','max']).reindex(times)
        np.testing.assert_array_equal(counts[:,j],other['count'].fillna(0).to_numpy())
        np.testing.assert_allclose(maximum[:,j],other['max'].to_numpy(),atol=1e-12,rtol=0,equal_nan=True)
        for h in np.flatnonzero(np.isnan(raw[:,j])):
            a=times[h]-pd.Timedelta(hours=1);b=times[h]
            frag=frame.loc[(frame.index>=a)&(frame.index<b)]
            sources=sorted(frag.source.dropna().unique().tolist());source_by_hour[f'{h}:{j}']=sources
            if np.isfinite(maximum[h,j]) and maximum[h,j]>10:
                exceed=frag[frag.value>10];witness=exceed.iloc[0]
                bad_rows.append(dict(hour_index=int(h),node_index=j,node=node,hour_start=a.isoformat(),hour_end=b.isoformat(),valid_count=int(counts[h,j]),partial_max=float(maximum[h,j]),witness_timestamp=exceed.index[0].isoformat(),witness_value=float(witness.value),witness_source=witness.source))
        print('AUDITED',node,flush=True)
    known_bad=np.isnan(raw)&np.isfinite(maximum)&(maximum>10)
    assert len(bad_rows)==known_bad.sum()
    np.savez_compressed(OUT/'partial_hour_evidence.npz',valid_count=counts,partial_max=maximum,known_bad=known_bad,timestamps=np.array([t.isoformat() for t in times]),nodes=nodes)
    pd.DataFrame(bad_rows).to_csv(OUT/'partial_exceedance_witnesses.csv',index=False,encoding='utf-8-sig')
    (OUT/'hour_source_files.json').write_text(json.dumps(source_by_hour,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(seen_files).to_csv(OUT/'raw_files.csv',index=False,encoding='utf-8-sig')
    for file,digest in identity.items():assert sha(file)==digest
    result=dict(status='PASS',source_files=len(seen_files),source_rows=sum(r['rows'] for r in seen_files),strict_unknown_hours=int(np.isnan(raw).sum()),partial_unknown_hours=int(((counts>0)&(counts<12)).sum()),partial_known_failures=int(known_bad.sum()),complete_hours_exactly_match=True,independent_partial_resample_matches=True,source_identity=identity,artifact_sha256=sha(OUT/'partial_hour_evidence.npz'))
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_identity'}),flush=True)

def requests():
    times,nodes,raw=load_truth();ev=np.load(OUT/'partial_hour_evidence.npz');counts=ev['valid_count'];maximum=ev['partial_max'];bad=ev['known_bad']
    sources=json.loads((OUT/'hour_source_files.json').read_text(encoding='utf-8'))
    old=ROOT/'artifacts/review_followup_v1/cross_execution/task_ledger.csv';new=ROOT/'artifacts/review_iteration2_v1/analog_execution/task_ledger.csv'
    frames=[]
    for p in [old,new]:
        d=pd.read_csv(p);d['experiment']='previous_cross' if p==old else 'analog_execution';frames.append(d)
    d=pd.concat(frames,ignore_index=True);unknown=d[d.status=='unknown']
    records={};physical_windows=set()
    for r in unknown.itertuples():
        node=int(r.node);start=int(r.issue+r.start+1);end=int(r.issue+r.end)
        window=(start,end,node);physical_windows.add(window)
        for h in range(start,end+1):
            if np.isfinite(raw[h,node]):continue
            key=(h,node)
            if key not in records:records[key]=dict(windows=set(),origins=set(),methods=set(),penalties=[])
            rec=records[key];rec['windows'].add(window);rec['origins'].add((r.fold,int(r.issue)));rec['methods'].add(str(r.prediction));rec['penalties'].append(float(r.penalty))
    out=[]
    for (h,j),rec in records.items():
        start=times[h]-pd.Timedelta(hours=1)
        out.append(dict(node=nodes[j],node_index=j,hour_index=h,hour_start=start.isoformat(),hour_end=times[h].isoformat(),field=FIELD,valid_5min_count=int(counts[h,j]),missing_or_invalid_5min_count=int(12-counts[h,j]),partial_max=float(maximum[h,j]) if np.isfinite(maximum[h,j]) else None,already_observed_exceedance=bool(bad[h,j]),affected_unique_windows=len(rec['windows']),affected_origins=len(rec['origins']),forecast_methods=';'.join(sorted(rec['methods'])),available_fragment_sources=';'.join(sources.get(f'{h}:{j}',[])),request='追溯来源与异常值；已有超限证据不必等待完整才能判失败' if bad[h,j] else '补交该小时12个5分钟原始值、质量码及时间戳；缺失保留未知'))
    result=pd.DataFrame(out).sort_values(['already_observed_exceedance','affected_unique_windows','affected_origins','node','hour_start'],ascending=[True,False,False,True,True]);result.insert(0,'request_order',np.arange(1,len(result)+1))
    result.to_csv(OUT/'targeted_missing_hours.csv',index=False,encoding='utf-8-sig')
    rows=[]
    unresolved=result[~result.already_observed_exceedance]
    for node,g in unresolved.groupby('node'):
        g=g.sort_values('hour_index');chunks=[];current=[];previous=None
        for r in g.itertuples():
            if previous is not None and r.hour_index!=previous+1:chunks.append(current);current=[]
            current.append(r);previous=r.hour_index
        if current:chunks.append(current)
        for chunk in chunks:rows.append(dict(node=node,start=chunk[0].hour_start,end=chunk[-1].hour_end,hours=len(chunk),missing_or_invalid_5min_count=sum(r.missing_or_invalid_5min_count for r in chunk),maximum_affected_unique_windows=max(r.affected_unique_windows for r in chunk)))
    pd.DataFrame(rows).sort_values(['maximum_affected_unique_windows','node','start'],ascending=[False,True,True]).to_csv(OUT/'targeted_missing_intervals.csv',index=False,encoding='utf-8-sig')
    summary=dict(status='PASS',related_strict_unknown_hours=len(result),already_observed_failures=int(result.already_observed_exceedance.sum()),unresolved_hours=len(unresolved),unresolved_intervals=len(rows),related_unique_task_windows=len(physical_windows),priority='number of affected unique fixed-action windows, not expected information value or strategy importance',task_ledgers={str(p):sha(p) for p in [old,new]},evidence_sha256=sha(OUT/'partial_hour_evidence.npz'))
    (OUT/'request_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','requests']);args=p.parse_args();globals()[args.mode]()
