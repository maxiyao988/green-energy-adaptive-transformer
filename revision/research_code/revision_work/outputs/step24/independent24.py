"""Independent numerical reference. Imports no repository/experiment module."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

def ranks(a):
 a=np.asarray(a,float);one=a.ndim==1
 if one:a=a[None,:]
 out=pd.DataFrame(a).rank(axis=1,method='average').to_numpy(float)
 return out[0] if one else out

def rank_ic(pred,y):
 x=ranks(np.atleast_2d(pred));z=ranks(np.atleast_2d(y));x-=np.mean(x,axis=1)[:,None];z-=np.mean(z,axis=1)[:,None]
 num=np.einsum('ij,ij->i',x,z);den=np.sqrt(np.einsum('ij,ij->i',x,x)*np.einsum('ij,ij->i',z,z));out=np.full(len(x),np.nan);np.divide(num,den,out=out,where=den>0);return out

def nw(x,lag):
 x=np.asarray(x,float);x=x[np.isfinite(x)];n=x.size;u=x-np.mean(x);ac=np.correlate(u,u,mode='full')[n-1:n+lag]/n;v=ac[0]+2*np.dot(1-np.arange(1,lag+1)/(lag+1),ac[1:]);se=np.sqrt(max(v,0)/n);mu=np.mean(x);t=mu/se if se>0 else np.nan
 return dict(mean=float(mu),se=float(se),t=float(t),p_two=float(2*norm.sf(abs(t))),p_one=float(norm.sf(t)),n=n)

def cw(y,p,b,shape,h):
 y=np.asarray(y,float);p=np.asarray(p,float);b=np.broadcast_to(np.asarray(b,float),y.shape);f=2*(y-b)*(p-b);date=f.reshape(shape).mean(axis=1);r=nw(date,h-1);return f,date,r

def holm(p):
 p=np.asarray(p,float);idx=np.argsort(p,kind='stable');z=np.maximum.accumulate(p[idx]*np.arange(len(p),0,-1));out=np.empty(len(p));out[idx]=np.minimum(z,1);return out

def metrics(y,p,shape,m):
 y=np.asarray(y,float);p=np.asarray(p,float);loss=(y-p)**2;ic=rank_ic(p.reshape(shape),y.reshape(shape));valid=y!=0;bm=(y-m)**2;zero=y*y
 return dict(mse=float(np.mean(loss)),rank_ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan,da=float(np.mean(np.sign(p[valid])==np.sign(y[valid]))),r2_train_mean=float(1-np.sum(loss)/np.sum(bm)),r2_zero=float(1-np.sum(loss)/np.sum(zero)),n_da=int(valid.sum())),loss.reshape(shape).mean(axis=1),ic

def mbb(T,L,B,rng):
 # Independently allocate each block into destination slots (not reshape-based).
 out=np.empty((B,T),dtype=np.int64);k=math.ceil(T/L);starts=rng.integers(0,T-L+1,size=(B,k))
 for b in range(k):
  n=min(L,T-b*L);out[:,b*L:b*L+n]=starts[:,b,None]+np.arange(n)
 return out

def joint_from_draws(x,seeds,dates,missing=False):
 x=np.asarray(x,float);per=np.stack([(np.nanmean if missing else np.mean)(x[i][dates],axis=1) for i in range(len(x))],axis=1);selected=np.take_along_axis(per,np.asarray(seeds),axis=1);return selected.mean(axis=1)

def pooled_nw(series,lag):
 ss=[np.asarray(x,float) for x in series];n=np.array([len(x) for x in ss]);w=n/n.sum();r=[nw(x,lag) for x in ss];mu=np.dot(w,[v['mean'] for v in r]);se=np.sqrt(np.dot(w*w,[v['se']**2 for v in r]));return dict(mean=float(mu),se=float(se),t=float(mu/se),p_two=float(2*norm.sf(abs(mu/se))),p_one=float(norm.sf(mu/se)))

def grs(excess,factor):
 # Standard finite-sample GRS with MLE covariance normalizations for residuals and factors.
 Y=np.asarray(excess,float);F=np.asarray(factor,float).reshape(len(Y),-1);T,N=Y.shape;K=F.shape[1];X=np.column_stack([np.ones(T),F]);coef=np.linalg.lstsq(X,Y,rcond=None)[0];res=Y-X@coef;S=res.T@res/T;v=(F-F.mean(0)).T@(F-F.mean(0))/T;a=coef[0];z=(T-N-K)/N*(a@np.linalg.solve(S,a))/(1+F.mean(0)@np.linalg.solve(v,F.mean(0)))
 from scipy.stats import f
 return float(z),float(f.sf(z,N,T-N-K))
