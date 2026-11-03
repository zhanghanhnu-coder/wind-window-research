"""Attach immutable round27 score paths to its original prediction-only inventory."""
from pathlib import Path

def attach_score_sources(items,root):
 rows=[]
 for item in items:
  p=Path(root)/item['fold']/f'scores_{item["model"]}_{item["seed"]}_{item["method"]}.npz'
  if 'score_source' in item and Path(item['score_source'])!=p:raise ValueError('existing score path conflicts with frozen round27 convention')
  if not p.is_file():raise FileNotFoundError(p)
  rows.append(dict(item,score_source=str(p)))
 return rows
