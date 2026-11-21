import pickle, numpy as np, pandas as pd, sys
from scipy.optimize import minimize
S='/tmp/claude-0/-home-user-uni/1abba573-8c9d-53cd-94f1-5aeee15bfbcb/scratchpad'
T1,T2='2026-09-01','2026-10-11'
R=pickle.load(open(f'{S}/step3_R.pkl','rb'))
VARS={'基本7':list(range(7)),'+着に強い力':list(range(10)),'+コンビ実績':list(range(7))+[10,11],'+レースの形':list(range(7))+[12,13],'全部':list(range(14))}
def stacks(rs,stage,cols):
    X=[];M=[];Y=[]
    for r in rs:
        i,j,k=r['top']
        if min(i,j,k)<0: continue
        n=r['n']; x=np.zeros((9,len(cols))); m=np.zeros(9,bool)
        if stage==2: x[:n]=r['X2'][i][:,cols]; m[:n]=True; m[i]=False; y=j
        else: x[:n]=r['X3'][i,j][:,cols]; m[:n]=True; m[[i,j]]=False; y=k
        X.append(x);M.append(m);Y.append(y)
    return np.array(X),np.array(M),np.array(Y)
def nll(w,X,M,Y,l2=1e-3):
    s=X@w; s=np.where(M,s,-1e9); mx=s.max(1,keepdims=True); lse=mx[:,0]+np.log(np.exp(s-mx).sum(1))
    return -(s[np.arange(len(Y)),Y]-lse).mean()+l2*(w[1:]**2).sum()
def fitw(rs,stage,cols):
    X,M,Y=stacks(rs,stage,cols); w0=np.zeros(len(cols)); w0[0]=1
    return minimize(nll,w0,args=(X,M,Y),method='L-BFGS-B').x
def combos(r,w2,w3,cols):
    n=r['n']; p=r['p']; cars=r['cars']
    s2=r['X2'][...,cols]@w2; s2[np.arange(n),np.arange(n)]=-np.inf
    q2=np.exp(s2-s2.max(1,keepdims=True)); q2/=q2.sum(1,keepdims=True)
    s3=r['X3'][...,cols]@w3
    for i in range(n):
        s3[i,:,i]=-np.inf; s3[:,i,i]=-np.inf
        for j in range(n): s3[i,j,j]=-np.inf
    mx=np.max(np.where(np.isfinite(s3),s3,-1e300),axis=2,keepdims=True)
    e3=np.exp(s3-mx); e3=np.nan_to_num(e3); q3=e3/np.maximum(e3.sum(2,keepdims=True),1e-300)
    Pm=p[:,None,None]*q2[:,:,None]*q3
    for i in range(n): Pm[i,i,:]=0
    out={'2車単':{},'2車複':{},'3連複':{},'ワイド':{}}
    P2=Pm.sum(2)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            out['2車単'][f"{cars[i]}>{cars[j]}"]=P2[i,j]
            k2='-'.join(map(str,sorted([cars[i],cars[j]]))); out['2車複'][k2]=out['2車複'].get(k2,0)+P2[i,j]
            for k in range(n):
                if k in (i,j): continue
                k3='-'.join(map(str,sorted([cars[i],cars[j],cars[k]]))); out['3連複'][k3]=out['3連複'].get(k3,0)+Pm[i,j,k]
    for k3,v in out['3連複'].items():
        a,b,cc=k3.split('-')
        for kk in (f"{a}-{b}",f"{a}-{cc}",f"{b}-{cc}"): out['ワイド'][kk]=out['ワイド'].get(kk,0)+v
    return out,q2
if __name__=='__main__':
    tr=[r for r in R if r['date']<T1]
    res={}
    for name,cols in VARS.items():
        w2=fitw(tr,2,cols); w3=fitw(tr,3,cols)
        res[name]=(cols,w2,w3)
        print(name,'2着係数',np.round(w2,3),'3着係数',np.round(w3,3),flush=True)
    pickle.dump(res,open(f'{S}/step3_w.pkl','wb'))
