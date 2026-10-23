import io, math, random, time, contextlib
import numpy as np
from BF import BF
from PPRL import Link

def _quiet(fn,*a,**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a,**kw)

def bf_to_np(bf,length):
    return np.unpackbits(np.frombuffer(bf.tobytes(),dtype=np.uint8))[:length]

def pair_features(x1,x2,mode="and_xor"):
    if mode=="and": return x1&x2
    if mode=="xor": return x1^x2
    if mode=="and_xor": return np.concatenate([x1&x2,x1^x2])
    if mode=="concat": return np.concatenate([x1,x2])
    raise ValueError(mode)

def prepare(size,corr,eps,L=1000,k=10,q=2,block=(2,),seed=0,data_dir="Datasets"):
    random.seed(seed)
    link=Link(L,k,q,0.8,[1,2,3,4],list(block),0,eps if eps is not None else 7)
    T={}
    t=time.perf_counter()
    db1=_quiet(link.read_database,f"{data_dir}/Alice_numrec_{size}_{corr}.csv")
    db2=_quiet(link.read_database,f"{data_dir}/Bob_numrec_{size}_{corr}.csv")
    T["read"]=time.perf_counter()-t
    t=time.perf_counter(); b1=_quiet(link.build_BI,db1); b2=_quiet(link.build_BI,db2); T["block"]=time.perf_counter()-t
    t=time.perf_counter(); bf1,v1=_quiet(link.data_encode,db1); bf2,v2=_quiet(link.data_encode,db2); T["encode"]=time.perf_counter()-t
    n=len(set(v1+v2)); fpr=(1-math.e**((-1*k*n)/L))**k
    t=time.perf_counter()
    if eps is None: p1,p2=bf1,bf2
    else: p1,p2=link.add_DP_noise(bf1),link.add_DP_noise(bf2)
    T["dp"]=time.perf_counter()-t
    cand=[(r1,r2) for blk in b1 if blk in b2 for r1 in b1[blk] for r2 in b2[blk]]
    return dict(link=link,db1=db1,db2=db2,bf1=p1,bf2=p2,cand=cand,fpr=fpr,T=T,L=L)

def _f1(tp,npred,ntrue):
    p=tp/npred if npred else 0.0; r=tp/ntrue if ntrue else 0.0
    return p,r,(2*p*r/(p+r) if (p+r) else 0.0)

def run_cv(P,models,feat="and_xor",K=5,seed=0,easy_neg_ratio=3,thr=0.8):
    rng=random.Random(seed)
    link,db1,db2,L=P["link"],P["db1"],P["db2"],P["L"]
    ids1={r:db1[r][0] for r in db1}; ids2={r:db2[r][0] for r in db2}
    ent_fold={}
    for e in ids1.values(): ent_fold.setdefault(e,rng.randrange(K))
    fold1={r:ent_fold[e] for r,e in ids1.items()}
    fold2={r:(ent_fold[e] if e in ent_fold else rng.randrange(K)) for r,e in ids2.items()}
    X1={r:bf_to_np(P["bf1"][r],L) for r in db1}; X2={r:bf_to_np(P["bf2"][r],L) for r in db2}
    cand=P["cand"]; cand_set=set(cand); all1=list(db1); all2=list(db2)
    tot={n:[0,0] for n in list(models)+["thr_fixed","thr_tuned"]}; ntrue_all=0
    times={n:0.0 for n in models}
    for f in range(K):
        train_c=[(a,b) for (a,b) in cand if fold1[a]!=f and fold2[b]!=f]
        test_c=[(a,b) for (a,b) in cand if fold1[a]==f and fold2[b]==f]
        tr1=[a for a in all1 if fold1[a]!=f]; tr2=[b for b in all2 if fold2[b]!=f]
        npos=sum(1 for a,b in train_c if ids1[a]==ids2[b])
        easy=set(); tries=0
        while len(easy)<easy_neg_ratio*max(npos,1) and tries<50*max(npos,1):
            a,b=rng.choice(tr1),rng.choice(tr2); tries+=1
            if ids1[a]!=ids2[b] and (a,b) not in cand_set: easy.add((a,b))
        train_pairs=train_c+list(easy)
        y_tr=np.array([1 if ids1[a]==ids2[b] else 0 for a,b in train_pairs])
        X_tr=np.array([pair_features(X1[a],X2[b],feat) for a,b in train_pairs],dtype=np.uint8)
        e_test={ids1[a] for a in all1 if fold1[a]==f}; e2_test={ids2[b] for b in all2 if fold2[b]==f}
        ntrue_all+=len(e_test&e2_test)
        if not test_c: continue
        y_te=np.array([1 if ids1[a]==ids2[b] else 0 for a,b in test_c])
        X_te=np.array([pair_features(X1[a],X2[b],feat) for a,b in test_c],dtype=np.uint8)
        for name,make in models.items():
            m=make(); t=time.perf_counter(); m.fit(X_tr,y_tr); pred=m.predict(X_te); times[name]+=time.perf_counter()-t
            tot[name][0]+=int(((pred==1)&(y_te==1)).sum()); tot[name][1]+=int((pred==1).sum())
        sim_te=np.array([link.bf.calc_bf_sim(P["bf1"][a],P["bf2"][b]) for a,b in test_c])
        sim_tr=np.array([link.bf.calc_bf_sim(P["bf1"][a],P["bf2"][b]) for a,b in train_pairs])
        p=sim_te>=thr; tot["thr_fixed"][0]+=int((p&(y_te==1)).sum()); tot["thr_fixed"][1]+=int(p.sum())
        best_t,best_f=thr,-1
        for t_ in np.arange(0.30,0.95,0.01):
            pp=sim_tr>=t_; tp_=int((pp&(y_tr==1)).sum()); _,_,ff=_f1(tp_,int(pp.sum()),int((y_tr==1).sum()))
            if ff>best_f: best_f,best_t=ff,t_
        p=sim_te>=best_t; tot["thr_tuned"][0]+=int((p&(y_te==1)).sum()); tot["thr_tuned"][1]+=int(p.sum())
    out={}
    for name,(tp,npred) in tot.items():
        pr,rc,f1=_f1(tp,npred,ntrue_all); out[name]=dict(prec=pr,rec=rc,f1=f1)
    out["_times"]=times; out["_ntrue"]=ntrue_all
    return out
