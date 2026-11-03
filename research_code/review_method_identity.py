"""Resolve historical aliases explicitly; never infer identity from B/W28 names."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
OUT=ART/'review_followup_v1/precision'

def main():
    inventory=ART/'state_gate_v1/inventory.json'
    entries=json.loads(inventory.read_text(encoding='utf-8'))
    names={'B':'Rolling28_Base49','W28':'Residual28'}
    rows=[]
    for row in entries:
        if row['case'] not in names:continue
        source=Path(row['q_source'])
        if row['case']=='B':assert source.parent.parent.name=='seasonal_rolling_v1' and source.stem.endswith('_rolling28')
        else:assert source.parent.parent.name=='residual_warm_v1' and source.stem.startswith('predictions_WT_')
        rows.append(dict(**row,canonical_name=names[row['case']],q_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ranks_sha256=hashlib.sha256(Path(row['ranks_source']).read_bytes()).hexdigest()))
    assert len(rows)==24
    result=dict(status='PASS',inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),aliases=names,
                correction='Precision diagnostic B/W28 are rolling affine versus residual-shape WT. They are not cross-execution fixed versus rolling affine.',
                precision_files=['fixed_forecast_scores64.csv','calibration_W28_minus_B.csv'],
                cross_execution_aliases={'B':'FixedMonth','W28':'Rolling28'},entries=rows)
    (OUT/'method_identity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 24 forecast source identities resolved')

if __name__=='__main__':main()
