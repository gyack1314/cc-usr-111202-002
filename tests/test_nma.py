"""网络合成的手算对照、独立求解交叉验证、不变性与敏感性检验。"""
import json
import math
import unittest
from pathlib import Path
from research_data.io import load
from research_data import nma

def _arm(r,n):
    return math.log(r)-math.log(n-r),1/r+1/(n-r)

def _matmul(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

def _transpose(a):
    return [list(c) for c in zip(*a)]

def _brute_inverse(a):
    """Gauss-Jordan求逆,与包内Cholesky路径完全独立。"""
    n=len(a);aug=[list(a[i])+[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv=max(range(col,n),key=lambda r:abs(aug[r][col]))
        aug[col],aug[piv]=aug[piv],aug[col];d=aug[col][col]
        aug[col]=[x/d for x in aug[col]]
        for r in range(n):
            if r!=col:
                f=aug[r][col];aug[r]=[x-f*y for x,y in zip(aug[r],aug[col])]
    return [row[n:] for row in aug]

def _brute_det(a):
    """部分主元消去求行列式,与包内Cholesky对数行列式独立。"""
    n=len(a);m=[list(row) for row in a];det=1.0
    for col in range(n):
        piv=max(range(col,n),key=lambda r:abs(m[r][col]))
        if piv!=col:m[col],m[piv]=m[piv],m[col];det=-det
        det*=m[col][col]
        for r in range(col+1,n):
            f=m[r][col]/m[col][col]
            for c in range(col,n):m[r][c]-=f*m[col][c]
    return det

def _normal_pieces():
    """手工重构normal.json的全部对比,不经过包内构造函数。"""
    lA,vA=_arm(12,40);lB,vB=_arm(20,40);lC,vC=_arm(18,36)
    lB2,vB2=_arm(22,44);lC2,vC2=_arm(15,42)
    lC3,vC3=_arm(17,50);lD3,vD3=_arm(25,50)
    y=[lB-lA,lC-lA,lC2-lB2,lD3-lC3]
    X=[[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,1.0,0.0],[0.0,-1.0,1.0]]
    blocks=[[[vB+vA,vA],[vA,vC+vA]],[[vC2+vB2]],[[vD3+vC3]]]
    return y,X,blocks

def _brute_fit(y,X,blocks,tau):
    """完整协方差一次性装配+暴力求逆的GLS与REML目标。"""
    k=len(y);p=len(X[0]);V=[[0.0]*k for _ in range(k)];off=0;t2=tau*tau
    for S in blocks:
        for i in range(len(S)):
            for j in range(len(S)):V[off+i][off+j]=S[i][j]+(t2 if i==j else 0.5*t2)
        off+=len(S)
    Vinv=_brute_inverse(V);Xt=_transpose(X)
    M=_matmul(Xt,_matmul(Vinv,X));Minv=_brute_inverse(M)
    beta=[sum(row) for row in _matmul(Minv,_matmul(Xt,_matmul(Vinv,[[v] for v in y])))]
    r=[y[i]-sum(X[i][j]*beta[j] for j in range(p)) for i in range(k)]
    q=sum(ri*vi for ri,vi in zip(r,[sum(row) for row in _matmul(Vinv,[[v] for v in r])]))
    obj=math.log(_brute_det(V))+math.log(_brute_det(M))+q
    return {'objective':obj,'beta':beta,'cov':Minv}

def _pairs_of(component):
    return {tuple(p['pair']):p for p in component['pairs']}

class NmaTests(unittest.TestCase):
    def test_saturated_hand_check(self):
        """单三臂研究:df=0强制tau=0,估计与手算完全一致。"""
        result=nma.synthesize(load('data/saturated-three-arm.json')[0])
        comp,=result['components']
        self.assertEqual(comp['treatments'],['A','B','C'])
        self.assertEqual(comp['residual_df'],0)
        self.assertTrue(comp['heterogeneity_unidentifiable'])
        self.assertEqual(comp['tau_selected'],0.0)
        self.assertEqual(len(comp['tau_grid_trace']),1)
        anchor=math.log(7/3)
        be={e['treatment']:e for e in comp['basic_effects']}
        self.assertAlmostEqual(be['B']['estimate'],anchor,places=12)
        self.assertAlmostEqual(be['C']['estimate'],anchor,places=12)
        pairs=_pairs_of(comp)
        self.assertAlmostEqual(pairs[('A','B')]['estimate'],-anchor,places=12)
        self.assertAlmostEqual(pairs[('B','C')]['estimate'],0.0,places=12)
        self.assertAlmostEqual(pairs[('B','C')]['se'],math.sqrt(0.1+1/9),places=12)
        self.assertAlmostEqual(pairs[('A','B')]['se'],math.sqrt(5/42+0.1),places=12)
        study,=comp['studies']
        self.assertAlmostEqual(study['weight_share'],1.0,places=12)
        for c in study['contrasts']:self.assertAlmostEqual(c['residual'],0.0,places=12)

    def test_disconnected_hand_check(self):
        """E-F分量df=0且强制tau=0;跨分量比较全部不可识别。"""
        result=nma.synthesize(load('data/disconnected.json')[0])
        comps={c['reference']:c for c in result['components']}
        ef=comps['E']
        self.assertEqual(ef['treatments'],['E','F'])
        self.assertEqual(ef['residual_df'],0)
        self.assertTrue(ef['heterogeneity_unidentifiable'])
        self.assertEqual(ef['tau_selected'],0.0)
        pair,=ef['pairs']
        self.assertEqual(pair['pair'],['E','F'])
        self.assertAlmostEqual(pair['estimate'],math.log(13/28),places=12)
        self.assertAlmostEqual(pair['se'],math.sqrt(1/4+1/16+1/7+1/13),places=12)
        cross={tuple(u['pair']) for u in result['unidentifiable_pairs']}
        expected={(a,b) for a in 'ABCD' for b in 'EF'}
        self.assertEqual(cross,expected)

    def test_normal_matches_brute_force(self):
        """normal.json每个网格tau的REML目标与选定拟合,对照独立暴力求解。"""
        value=load('data/normal.json')[0]
        result=nma.synthesize(value)
        comp,=result['components']
        self.assertEqual(comp['residual_df'],1)
        y,X,blocks=_normal_pieces()
        brutes={t:_brute_fit(y,X,blocks,t) for t in value['tau_grid']}
        for row in comp['tau_grid_trace']:
            self.assertAlmostEqual(row['objective'],brutes[row['tau']]['objective'],places=9)
        tol=value['solver']['tolerance']
        best=value['tau_grid'][0]
        for t in value['tau_grid'][1:]:
            if brutes[t]['objective']<brutes[best]['objective']-tol:best=t
        self.assertEqual(comp['tau_selected'],best)
        sel=brutes[comp['tau_selected']]
        be={e['treatment']:e for e in comp['basic_effects']}
        for i,t in enumerate(['B','C','D']):
            self.assertAlmostEqual(be[t]['estimate'],sel['beta'][i],places=9)
            self.assertAlmostEqual(be[t]['se'],math.sqrt(sel['cov'][i][i]),places=9)
        pairs=_pairs_of(comp)
        d={'A':0.0,'B':sel['beta'][0],'C':sel['beta'][1],'D':sel['beta'][2]}
        def cov(a,b):
            idx={'B':0,'C':1,'D':2}
            if a=='A' or b=='A':return 0.0
            return sel['cov'][idx[a]][idx[b]]
        for (a,b),p in pairs.items():
            self.assertAlmostEqual(p['estimate'],d[a]-d[b],places=9)
            self.assertAlmostEqual(p['se'],math.sqrt(cov(a,a)+cov(b,b)-2*cov(a,b)),places=9)
            self.assertAlmostEqual(p['interval'][0],p['estimate']-value['normal_z']*p['se'],places=9)

    def test_arm_and_study_order_invariance(self):
        """更换臂顺序与研究顺序不改变可识别治疗对结果。"""
        value=load('data/normal.json')[0]
        base=nma.synthesize(value)
        shuffled=json.loads(json.dumps(value))
        for s in shuffled['studies']:s['arms'].reverse()
        shuffled['studies'].reverse()
        alt=nma.synthesize(shuffled)
        for c0,c1 in zip(base['components'],alt['components']):
            self.assertEqual(c0['tau_selected'],c1['tau_selected'])
            for p0,p1 in zip(c0['pairs'],c1['pairs']):
                self.assertEqual(p0['pair'],p1['pair'])
                self.assertAlmostEqual(p0['estimate'],p1['estimate'],places=10)
                self.assertAlmostEqual(p0['se'],p1['se'],places=10)

    def test_reference_rule_invariance(self):
        """局部参考重选是线性坐标变换,不改变两两比较与tau选择。"""
        value=load('data/normal.json')[0]
        base=nma.synthesize(value)
        alt=nma.synthesize(value,reference_rule=lambda arms:len(arms)-1)
        for c0,c1 in zip(base['components'],alt['components']):
            self.assertEqual(c0['tau_selected'],c1['tau_selected'])
            for p0,p1 in zip(c0['pairs'],c1['pairs']):
                self.assertAlmostEqual(p0['estimate'],p1['estimate'],places=9)
                self.assertAlmostEqual(p0['se'],p1['se'],places=9)

    def test_leave_one_out_vanishing(self):
        """删除桥接研究后D的比较如实消失;删除S1后A消失且df=0。"""
        value=load('data/normal.json')[0]
        runs={r['removed']:r for r in nma.leave_one_out(value)}
        gone=runs['bridge']
        self.assertEqual(gone['vanished_pairs'],[['A','D'],['B','D'],['C','D']])
        comp,=gone['synthesis']['components']
        self.assertEqual(comp['treatments'],['A','B','C'])
        s1=runs['S1']
        self.assertEqual(s1['vanished_pairs'],[['A','B'],['A','C'],['A','D']])
        comp,=s1['synthesis']['components']
        self.assertEqual(comp['treatments'],['B','C','D'])
        self.assertEqual(comp['residual_df'],0)
        self.assertTrue(comp['heterogeneity_unidentifiable'])
        self.assertEqual(comp['tau_selected'],0.0)
        s2=runs['S2']
        self.assertEqual(s2['vanished_pairs'],[])
        self.assertEqual(s2['synthesis']['components'][0]['residual_df'],0)

    def test_tau_tie_break(self):
        """同分在容差内取较小tau,超出容差才更新。"""
        fits=[{'tau':0.0,'objective':2.0},{'tau':0.1,'objective':2.0-5e-10},{'tau':0.2,'objective':3.0}]
        self.assertEqual(nma._select_tau(fits,1e-9)['tau'],0.0)
        fits=[{'tau':0.0,'objective':2.0},{'tau':0.1,'objective':2.0-2e-9}]
        self.assertEqual(nma._select_tau(fits,1e-9)['tau'],0.1)

    def test_weights_sum_to_parameters(self):
        """逐研究杠杆之和等于分量参数数,权重份额归一。"""
        for f in ('data/normal.json','data/disconnected.json'):
            result=nma.synthesize(load(f)[0])
            for comp in result['components']:
                self.assertAlmostEqual(sum(s['leverage'] for s in comp['studies']),comp['parameters'],places=9)
                self.assertAlmostEqual(sum(s['weight_share'] for s in comp['studies']),1.0,places=9)
                self.assertGreaterEqual(comp['information_condition'],1.0)
                self.assertLess(comp['solve_residual'],1e-6)

    def test_validation_errors(self):
        """端点计数与非法tau网格被拒绝。"""
        with self.assertRaises(ValueError):nma.synthesize(load('data/invalid-domain.json')[0])
        value=load('data/normal.json')[0]
        for bad in ([0.1,0.1],[0.2,0.05],[-0.1,0.5],[]):
            broken=json.loads(json.dumps(value));broken['tau_grid']=bad
            with self.assertRaises(ValueError):nma.synthesize(broken)

if __name__=='__main__':unittest.main()
