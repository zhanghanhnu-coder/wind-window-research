"""Correct unverified physical-unit wording in iteration-three deliverables only."""
from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_iteration3_v1/acquisition'
DOC=ROOT.parent/'研究笔记/第三轮实证推进_2026-11-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
targets=[ROOT/'iteration3_acquisition_run.py',OUT/'essential_hours.csv',DOC/'补录输出字段说明.md',DOC/'补录决策实验.md']
before={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p not in targets and p.name not in ['source_sha256.json','unit_wording_verification.json']}
numeric_before=pd.read_csv(OUT/'essential_hours.csv',dtype={'node':str}).drop(columns='request')
old_unit='10'+'m'+'/s'
changed=[]
for p in targets:
    raw=p.read_bytes();bom=raw.startswith(b'\xef\xbb\xbf');s=raw.decode('utf-8-sig')
    s=s.replace('超过'+old_unit+'阈值','超过原始量纲阈值10（单位待确认）')
    s=s.replace('超'+old_unit+'阈值','超过原始量纲阈值10（单位待确认）')
    s=s.replace('<='+old_unit,'<=原始量纲阈值10（单位待确认）')
    assert old_unit not in s
    new=(b'\xef\xbb\xbf' if bom else b'')+s.encode('utf-8')
    if new!=raw:p.write_bytes(new);changed.append(str(p))
numeric_after=pd.read_csv(OUT/'essential_hours.csv',dtype={'node':str}).drop(columns='request')
pd.testing.assert_frame_equal(numeric_before,numeric_after,check_exact=True)
assert all(sha(Path(p))==digest for p,digest in before.items())
sources=json.loads((OUT/'source_sha256.json').read_text(encoding='utf-8'))
for p in [ROOT/'iteration3_acquisition_run.py',Path(__file__).resolve()]:sources[str(p)]=sha(p)
(OUT/'source_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
result=dict(status='PASS',changed_text_files=changed,wording='原始量纲阈值10（单位待确认）',essential_hour_numeric_columns_exactly_unchanged=True,untouched_artifact_sha256_verified=len(before),all_cost_results_plans_and_certificates_unchanged=True,source_hash_manifest_updated=True)
(OUT/'unit_wording_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
