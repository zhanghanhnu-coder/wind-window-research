"""Resume frozen round35 through an explicit inventory adapter; preserve prior artifacts."""
from pathlib import Path
import research_rolling_copula as r
from rolling_copula_source_adapter import attach_score_sources

if __name__=='__main__':
 r.guard();r.prepare();checkpoint=r.OUT/'source_adapter_checkpoint.json'
 if checkpoint.exists():saved=r.read(checkpoint)
 else:
  saved={str(p):r.sha(p) for p in r.OUT.rglob('*') if p.is_file() and not p.name.startswith('source_adapter_')};r.js(checkpoint,saved)
 for file,h in saved.items():assert r.sha(Path(file))==h,file
 sources=[Path(__file__),Path(__file__).with_name('rolling_copula_source_adapter.py'),Path(__file__).parent/'tests/test_rolling_copula_source_adapter.py',checkpoint];identity={str(p):r.sha(p) for p in sources};dest=r.OUT/'source_adapter_identity.json'
 if dest.exists():assert r.read(dest)==identity
 else:r.js(dest,identity)
 original_read=r.read
 def adapted_read(path):
  value=original_read(path)
  return attach_score_sources(value,r.ROLLING) if Path(path)==r.ROLLING/'inventory.json' else value
 r.read=adapted_read;r.templates()
 for file,h in saved.items():assert r.sha(Path(file))==h,file
 r.js(r.OUT/'source_adapter_verification.json',dict(status='PASS',previous_files_unchanged=len(saved),adapter_identity_files=len(identity),correction='attach canonical frozen round27 score paths; no forecast or fitting changes'));r.evaluate()
