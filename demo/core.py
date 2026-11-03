"""Synthetic public demonstration. No observed data or fitted model."""
import math
def truth(values,threshold=10):
 if any(v is not None and v>threshold for v in values):return 0
 if all(v is not None for v in values):return 1
 return None
def probability(paths,start=0,duration=6,threshold=10):
 if not paths or duration<=0 or any(len(p)<start+duration for p in paths):raise ValueError('Incomplete scenarios')
 return sum(all(v<=threshold for v in p[start:start+duration]) for p in paths)/len(paths)
def scenarios(wind=8,n=57):
 return [[max(0,wind+2*math.sin(h/10+s*.13)+1.4*math.cos(h/19+s*.31)) for h in range(72)] for s in range(n)]
def snapshot(wind=8,threshold=10,crews=2):
 paths=scenarios(wind);windows=[{'start':h,'probability':probability(paths,h,6,threshold)} for h in [0,24,48,66]]
 selected=[w['start'] for w in windows if w['probability']>=.8][:crews]
 return {'synthetic':True,'site':'xx风电场','windows':windows,'selected':selected,'policy':'Illustrative threshold-and-capacity rule, NOT the research optimizer','mean':[sum(p[h] for p in paths)/len(paths) for h in range(72)]}
