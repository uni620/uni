import numpy as np
def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
