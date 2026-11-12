"""一致性相对效应的网络合成:多臂相关GLS、网格REML与逐研究删除敏感性。

每研究以treatment_id排序最小臂为局部参考,构造其余臂减参考臂的log-odds;
臂抽样方差1/r+1/(n-r),比较向量协方差对角为本臂加参考臂方差、非对角为参考臂方差,
研究间独立,多臂相关矩阵保持完整,不对角化。每连通分量单独拟合,分量内最小
treatment_id为零参考,跨分量比较不可识别。tau在预设非负升序网格上最小化REML目标
log|V|+log|X^T V^-1 X|+加权残差平方,同分在容差内取较小tau;残差自由度为零时仅给
tau=0条件拟合并标记异质性不可识别。数值上只用Cholesky/Jacobi分解求解并公开秩与
病态程度,不用裸矩阵求逆或岭项。对比方向统一为排序在前治疗减排序在后治疗。
"""
import math
from .io import finite,integer,positive,unique
from . import linalg

def _validated_arms(study):
    """核对计数并按treatment_id升序返回带log-odds与方差的臂列表。"""
    arms=study['arms'];unique(arms,'treatment_id')
    if len(arms)<2:raise ValueError('研究需有至少两个臂')
    out=[]
    for a in sorted(arms,key=lambda x:x['treatment_id']):
        n=integer(a['total'],1);r=integer(a['responders'],1)
        if r>=n:raise ValueError('计数不允许端点')
        out.append({'treatment_id':a['treatment_id'],'log_odds':math.log(r)-math.log(n-r),'variance':1/r+1/(n-r)})
    return out

def _contrast_block(arms,ref_index):
    """非参考臂减参考臂的log-odds及抽样协方差(对角v_k+v_0,非对角v_0)。"""
    ref=arms[ref_index];others=arms[:ref_index]+arms[ref_index+1:]
    y=[a['log_odds']-ref['log_odds'] for a in others]
    v0=ref['variance'];k=len(others)
    S=[[(others[i]['variance'] if i==j else 0.0)+v0 for j in range(k)] for i in range(k)]
    return others,y,S

def _with_heterogeneity(S,tau):
    """附加每研究异质性协方差:对角tau^2,非对角tau^2/2。"""
    k=len(S);t2=tau*tau
    return [[S[i][j]+(t2 if i==j else 0.5*t2) for j in range(k)] for i in range(k)]

def _components(studies):
    """以研究臂为边求治疗的连通分量,返回按最小ID排序的分量列表。"""
    parent={}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for s in studies:
        ts=[a['treatment_id'] for a in s['arms']]
        for t in ts:
            if t not in parent:parent[t]=t
        for t in ts[1:]:
            ra,rb=find(ts[0]),find(t)
            if ra!=rb:parent[rb]=ra
    groups={}
    for t in parent:groups.setdefault(find(t),[]).append(t)
    return sorted((sorted(g) for g in groups.values()),key=lambda g:g[0])

def _tau_grid(raw):
    if not isinstance(raw,list) or not raw:raise ValueError('tau_grid需要非空列表')
    g=[float(finite(t)) for t in raw]
    for t in g:
        if t<0.0:raise ValueError('tau_grid需非负')
    if any(g[i+1]<=g[i] for i in range(len(g)-1)):raise ValueError('tau_grid需严格升序')
    return g

def _gls(blocks,p,tau,tol,max_sweeps):
    """给定tau的完整协方差GLS:参数、条件协方差、REML目标各项与逐研究量。"""
    M=[[0.0]*p for _ in range(p)];g=[0.0]*p;logV=0.0;prepared=[]
    for b in blocks:
        V=_with_heterogeneity(b['S'],tau);L=linalg.cholesky(V);logV+=linalg.logdet(L)
        ViX=linalg.solve_matrix(L,b['X']);Viy=linalg.solve(L,b['y'])
        for r in range(p):
            g[r]+=sum(b['X'][i][r]*Viy[i] for i in range(len(Viy)))
            for c in range(r,p):M[r][c]+=sum(b['X'][i][r]*ViX[i][c] for i in range(len(Viy)))
        prepared.append((b,L,V,ViX))
    for r in range(p):
        for c in range(r):M[r][c]=M[c][r]
    eigs=linalg.jacobi_eigenvalues(M,max_sweeps);lo,hi=eigs[0],eigs[-1]
    if lo<=hi*tol:raise ArithmeticError('信息矩阵秩不足或病态,拒绝用岭项掩盖')
    LM=linalg.cholesky(M);beta=linalg.solve(LM,g);logM=linalg.logdet(LM)
    Minv=linalg.solve_matrix(LM,linalg.identity(p))
    res=max((abs(sum(M[i][j]*beta[j] for j in range(p))-g[i]) for i in range(p)),default=0.0)
    q=0.0;details=[]
    for b,L,V,ViX in prepared:
        r=[b['y'][i]-sum(b['X'][i][j]*beta[j] for j in range(p)) for i in range(len(b['y']))]
        Vr=linalg.solve(L,r);q+=sum(r[i]*Vr[i] for i in range(len(r)))
        G=[[sum(b['X'][i][a]*ViX[i][c] for i in range(len(ViX))) for c in range(p)] for a in range(p)]
        lev=sum(G[a][c]*Minv[a][c] for a in range(p) for c in range(p))
        details.append({'study_id':b['study_id'],'labels':b['labels'],'residuals':r,
                        'marginal_sd':[math.sqrt(V[i][i]) for i in range(len(r))],'leverage':lev})
    return {'tau':tau,'objective':logV+logM+q,'logdet_v':logV,'logdet_m':logM,'weighted_ssr':q,
            'beta':beta,'cov':Minv,'condition':hi/lo,'solve_residual':res,'studies':details}

def _select_tau(fits,tol):
    """网格上取REML目标最小者;同分(容差tol内)保留较小tau,网格需升序。"""
    best=fits[0]
    for f in fits[1:]:
        if f['objective']<best['objective']-tol:best=f
    return best

def _fit_component(studies,treatments,taus,tol,max_sweeps,z,reference_rule):
    """单连通分量:构造比较、网格REML选tau、方向一致的两两比较与逐研究证据。"""
    ref=treatments[0];cols={t:i for i,t in enumerate(treatments[1:])};p=len(cols)
    blocks=[]
    for s in studies:
        arms=_validated_arms(s)
        ri=reference_rule(arms) if reference_rule else 0
        others,y,S=_contrast_block(arms,ri);ref_t=arms[ri]['treatment_id']
        X=[]
        for a in others:
            row=[0.0]*p
            if a['treatment_id']!=ref:row[cols[a['treatment_id']]]=1.0
            if ref_t!=ref:row[cols[ref_t]]-=1.0
            X.append(row)
        blocks.append({'study_id':s['study_id'],'labels':['%s-%s'%(a['treatment_id'],ref_t) for a in others],
                       'y':y,'S':S,'X':X})
    k=sum(len(b['y']) for b in blocks);df=k-p
    if df<0:raise ArithmeticError('比较数少于参数,设计异常')
    if df==0:
        fits=[_gls(blocks,p,0.0,tol,max_sweeps)];selected=fits[0];unident=True
        tau_note='残差自由度为零,仅给tau=0条件拟合,异质性不可识别'
    else:
        fits=[_gls(blocks,p,t,tol,max_sweeps) for t in taus];selected=_select_tau(fits,tol)
        unident=False;tau_note='网格REML最优,同分在容差内取较小tau'
    beta=selected['beta'];cov=selected['cov']
    d={ref:0.0}
    for i,t in enumerate(treatments[1:]):d[t]=beta[i]
    def cov_of(a,b):
        if a==ref or b==ref:return 0.0
        return cov[cols[a]][cols[b]]
    pairs=[]
    for i in range(len(treatments)):
        for j in range(i+1,len(treatments)):
            a,b=treatments[i],treatments[j]
            est=d[a]-d[b];se=math.sqrt(cov_of(a,a)+cov_of(b,b)-2.0*cov_of(a,b))
            row={'pair':[a,b],'estimate':est,'se':se}
            if z is not None:row['interval']=[est-z*se,est+z*se]
            pairs.append(row)
    studies_out=[]
    for sd in selected['studies']:
        contrasts=[{'contrast':lab,'residual':r,'standardized':r/msd}
                   for lab,r,msd in zip(sd['labels'],sd['residuals'],sd['marginal_sd'])]
        studies_out.append({'study_id':sd['study_id'],'contrasts':contrasts,
                            'leverage':sd['leverage'],'weight_share':sd['leverage']/p})
    return {'treatments':treatments,'reference':ref,'parameters':p,'contrasts':k,'residual_df':df,
            'heterogeneity_unidentifiable':unident,'tau_note':tau_note,'tau_selected':selected['tau'],
            'tau_grid_trace':[{'tau':f['tau'],'objective':f['objective'],'logdet_v':f['logdet_v'],
                               'logdet_m':f['logdet_m'],'weighted_ssr':f['weighted_ssr'],
                               'information_condition':f['condition']} for f in fits],
            'information_condition':selected['condition'],'solve_residual':selected['solve_residual'],
            'basic_effects':[{'treatment':t,'estimate':d[t],'se':math.sqrt(cov_of(t,t))} for t in treatments[1:]],
            'pairs':pairs,'studies':studies_out}

def _method_notes(tol,taus,z):
    return {'contrast':'每研究以treatment_id最小臂为局部参考,其余臂减参考臂的log-odds;臂抽样方差1/r+1/(n-r);'
                       '比较向量协方差对角为本臂加参考臂方差、非对角为参考臂方差;研究间独立,多臂相关矩阵保持完整不对角化',
            'heterogeneity':'每研究附加协方差对角tau^2、非对角tau^2/2;tau在预设非负升序网格上最小化REML目标'
                            'log|V|+log|X^T V^-1 X|+加权残差平方(省略与tau无关常数),同分在容差内取较小tau;'
                            '报告网格最优而非连续最优',
            'identifiability':'每个连通分量单独拟合一致性相对效应,分量内最小treatment_id为零参考;跨分量比较不可识别;'
                              '信息矩阵经特征值检测秩与病态,不使用裸矩阵求逆或岭项',
            'interval':('区间为正态近似,以选定tau为条件,未计入tau选择不确定性' if z is not None
                        else '未提供normal_z,不给出区间'),
            'solver_tolerance':tol,'tau_grid':taus,'normal_z':z}

def synthesize(value,reference_rule=None):
    """完整网络合成:分量拟合、方向一致的两两比较与不可识别对照清单。"""
    studies=value['studies'];unique(studies,'study_id')
    taus=_tau_grid(value['tau_grid']);solver=value.get('solver',{})
    tol=positive(solver.get('tolerance',1e-9));max_sweeps=integer(solver.get('max_iterations',10000),1)
    z=value.get('normal_z')
    if z is not None:z=positive(z)
    comps=_components(studies);comp_of={}
    for idx,ts in enumerate(comps):
        for t in ts:comp_of[t]=idx
    buckets=[[] for _ in comps]
    for s in studies:buckets[comp_of[s['arms'][0]['treatment_id']]].append(s)
    out=[_fit_component(ss,ts,taus,tol,max_sweeps,z,reference_rule) for ts,ss in zip(comps,buckets)]
    unident=[{'pair':[a,b],'reason':'跨连通分量,不可识别'}
             for i in range(len(comps)) for j in range(i+1,len(comps))
             for a in comps[i] for b in comps[j]]
    return {'method':_method_notes(tol,taus,z),'components':out,'unidentifiable_pairs':unident}

def leave_one_out(value):
    """逐研究删除后重新建图并重选tau;断开的比较在vanished_pairs中如实列出。"""
    full=synthesize(value)
    full_pairs={tuple(p['pair']) for c in full['components'] for p in c['pairs']}
    runs=[]
    for s in value['studies']:
        sid=s['study_id'];reduced=dict(value)
        reduced['studies']=[x for x in value['studies'] if x['study_id']!=sid]
        r=synthesize(reduced)
        r_pairs={tuple(p['pair']) for c in r['components'] for p in c['pairs']}
        runs.append({'removed':sid,'vanished_pairs':[list(p) for p in sorted(full_pairs-r_pairs)],'synthesis':r})
    return runs
