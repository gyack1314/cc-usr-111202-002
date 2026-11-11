"""逐臂核对二元计数并展示观察对数优势。"""
import math
from .io import unique,integer

def summarize(v):
    unique(v['studies'],'study_id');result=[]
    for s in v['studies']:
        arms=s['arms'];unique(arms,'treatment_id')
        if len(arms)<2:raise ValueError('研究需有至少两个臂')
        for a in sorted(arms,key=lambda x:x['treatment_id']):
            n=integer(a['total'],1);r=integer(a['responders'],1)
            if r>=n:raise ValueError('计数不允许端点')
            result.append({'study_id':s['study_id'],'treatment_id':a['treatment_id'],'log_odds':math.log(r)-math.log(n-r),'sampling_variance':1/r+1/(n-r)})
    return result
