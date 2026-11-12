"""小维对称矩阵的分解求解:Cholesky因子、前代回代与循环Jacobi特征值。

只提供分解与求解,不提供裸矩阵求逆接口;秩与病态程度经特征值公开。
矩阵按行存储为list[list[float]]。
"""
import math

def cholesky(a):
    """返回下三角L使 a = L L^T;a非对称正定时抛ArithmeticError,不添加岭项。"""
    n=len(a);L=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=a[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j:
                if s<=0.0:raise ArithmeticError('矩阵非正定,疑似秩不足或断网')
                L[i][j]=math.sqrt(s)
            else:L[i][j]=s/L[j][j]
    return L

def logdet(factor):
    """由Cholesky因子求对数行列式。"""
    return 2.0*sum(math.log(factor[i][i]) for i in range(len(factor)))

def solve(factor,b):
    """对 L L^T x = b 前代回代求解,b为向量,返回新向量。"""
    n=len(factor);y=[0.0]*n
    for i in range(n):y[i]=(b[i]-sum(factor[i][k]*y[k] for k in range(i)))/factor[i][i]
    x=[0.0]*n
    for i in range(n-1,-1,-1):x[i]=(y[i]-sum(factor[k][i]*x[k] for k in range(i+1,n)))/factor[i][i]
    return x

def solve_matrix(factor,b):
    """逐列求解 L L^T X = B,b按行存储,返回同形矩阵。"""
    n=len(b);m=len(b[0])
    cols=[[b[i][j] for i in range(n)] for j in range(m)]
    out=[solve(factor,c) for c in cols]
    return [[out[j][i] for j in range(m)] for i in range(n)]

def identity(n):
    return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

def jacobi_eigenvalues(a,max_sweeps=10000,tol=1e-14):
    """循环Jacobi求对称矩阵特征值,返回升序列表,用于秩与病态检测。"""
    n=len(a);A=[row[:] for row in a]
    for _ in range(max(1,int(max_sweeps))):
        off=math.sqrt(sum(A[i][j]*A[i][j] for i in range(n) for j in range(i+1,n)))
        if off<=tol:break
        for p in range(n-1):
            for q in range(p+1,n):
                apq=A[p][q]
                if apq==0.0:continue
                theta=(A[q][q]-A[p][p])/(2.0*apq)
                sign=1.0 if theta>=0.0 else -1.0
                t=sign/(abs(theta)+math.hypot(theta,1.0))
                c=1.0/math.hypot(t,1.0);s=t*c
                for k in range(n):
                    akp=A[k][p];akq=A[k][q]
                    A[k][p]=c*akp-s*akq;A[k][q]=s*akp+c*akq
                for k in range(n):
                    apk=A[p][k];aqk=A[q][k]
                    A[p][k]=c*apk-s*aqk;A[q][k]=s*apk+c*aqk
    return sorted(A[i][i] for i in range(n))
