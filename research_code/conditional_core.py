"""Fixed conditional forecast designs; linear predictive information, not causal discovery."""
import numpy as np
import pandas as pd
from event_graph import angle_delta

LAGS=[0,1,2,3,6,12]

def shift(a,lag):
 a=np.asarray(a,dtype=float);out=np.full_like(a,np.nan)
 if lag==0:return a.copy()
 if abs(lag)>=len(a):return out
 if lag>0:out[lag:]=a[:-lag]
 else:out[:lag]=a[-lag:]
 return out

def changes(a,angular=False):return angle_delta(a,shift(a,1)) if angular else np.asarray(a)-shift(a,1)

def common_change(values,source,target):
 columns=[k for k in range(values.shape[1]) if k not in [source,target]];z=pd.DataFrame(values[:,columns])
 return z.median(axis=1).where(z.count(axis=1)>=10).to_numpy()

def error(predicted_delta,actual_delta,signal,current=None):
 if signal=='direction':return abs(angle_delta(np.clip(predicted_delta,-180,180),actual_delta))
 if current is None:return abs(np.asarray(predicted_delta)-actual_delta)
 return abs(np.clip(current+predicted_delta,0,60)-(current+actual_delta))

def formula_error(north,heading,relative,sign,offset):return abs(angle_delta(north,heading+sign*relative+offset))

def design(wind,direction,source,target,signal):
 dw=changes(wind);dd=changes(direction,True);series=dw if signal=='wind' else dd
 hour=2*np.pi*(np.arange(len(wind))%288)/288
 own=np.column_stack([shift(series[:,target],lag) for lag in LAGS]+[wind[:,target],np.sin(np.deg2rad(direction[:,target])),np.cos(np.deg2rad(direction[:,target])),np.sin(hour),np.cos(hour)])
 common=np.column_stack([shift(common_change(values,source,target),lag) for values in [dw,dd] for lag in LAGS])
 features=np.column_stack([own,common,series[:,source],shift(series[:,source],5),shift(series[:,source],2016)])
 return {'features':features,'target':shift(series[:,target],-1),'current':np.asarray(wind[:,target]),'columns':{'own':list(range(11)),'common':list(range(23)),'source5':list(range(23))+[23],'source30':list(range(23))+[24],'source_week':list(range(23))+[25]}}
