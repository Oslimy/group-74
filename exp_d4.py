import warnings, time, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from ml_linkage import prepare, run_cv

MODELS={
 "LR": lambda: LogisticRegression(max_iter=2000,class_weight="balanced"),
 "RF": lambda: RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=-1,random_state=0),
 "SVM": lambda: LinearSVC(class_weight="balanced",max_iter=5000),
 "XGB": lambda: XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.2,n_jobs=-1,eval_metric="logloss",verbosity=0),
}
METHODS=["thr_fixed","thr_tuned","LR","RF","SVM","XGB"]
C={"thr_fixed":"#C44536","thr_tuned":"#E09F3E","LR":"#1F3A5F","RF":"#2E7D6B","SVM":"#7B4BA8","XGB":"#4A90A4"}
LAB={"thr_fixed":"Threshold 0.8 (baseline)","thr_tuned":"Threshold (tuned)","LR":"Logistic Regression","RF":"Random Forest","SVM":"SVM (linear)","XGB":"XGBoost"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"figure.dpi":150})
def style(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

EPS=[("inf",None),("10",10),("7",7),("5",5),("4",4),("3",3),("2",2),("1",1)]
SEEDS=2
rows=[]; t0=time.time()
data={}
for size in (100,500):
    for lab,eps in EPS:
        acc={m:[] for m in METHODS}; pr={m:[] for m in METHODS}; rc={m:[] for m in METHODS}; tm={m:[] for m in MODELS}
        for s in range(SEEDS):
            P=prepare(size,"corr_50",eps,seed=s); r=run_cv(P,MODELS,seed=s)
            for m in METHODS: acc[m].append(r[m]["f1"]); pr[m].append(r[m]["prec"]); rc[m].append(r[m]["rec"])
            for m in MODELS: tm[m].append(r["_times"][m])
        for m in METHODS:
            rows.append(dict(size=size,corr="corr_50",eps=lab,model=m,f1=np.mean(acc[m]),prec=np.mean(pr[m]),rec=np.mean(rc[m]),
                             secs=(np.mean(tm[m]) if m in MODELS else None),fpr=P["fpr"]))
        data[(size,lab)]={m:round(float(np.mean(acc[m])),3) for m in METHODS}
    print(f"size {size} done ({time.time()-t0:.0f}s)",flush=True)

df=pd.DataFrame(rows); df.to_csv("all_results_d4.csv",index=False)
xl=[l for l,_ in EPS]
def plot_eps(size,fn,title):
    sub=df[df["size"]==size]; fig,ax=plt.subplots(figsize=(7.4,4.5)); x=np.arange(len(xl))
    for m in METHODS:
        y=[sub[(sub.eps==l)&(sub.model==m)].f1.values[0] for l in xl]
        ax.plot(x,y,marker="o",ms=4.5,lw=2,color=C[m],label=LAB[m])
    ax.set_xticks(x); ax.set_xticklabels(["∞" if l=="inf" else l for l in xl])
    ax.set_xlabel("Privacy budget ε  (left = no noise, right = strong privacy)"); ax.set_ylabel("F1 score")
    ax.set_ylim(-0.03,1.03); ax.set_title(title,fontweight="bold"); ax.legend(frameon=False,fontsize=8.5,ncol=2,loc="lower left"); style(ax)
    fig.tight_layout(); fig.savefig(f"figs/{fn}",bbox_inches="tight"); plt.close(fig)
plot_eps(500,"d4_f1_500.png","Linkage quality vs privacy budget — all 5 models (500 records, 50% corruption)")
plot_eps(100,"d4_f1_100.png","Linkage quality vs privacy budget — all 5 models (100 records, 50% corruption)")

# corruption comparison at eps=7 for the 4 models + tuned threshold
crows=[]
for corr in ("non_corr","corr_25","corr_50"):
    for m in ["thr_tuned","LR","RF","SVM","XGB"]:
        vals=[]
        for s in range(SEEDS):
            P=prepare(500,corr,7,seed=s); r=run_cv(P,MODELS,seed=s); vals.append(r[m]["f1"])
        crows.append(dict(corr=corr,model=m,f1=float(np.mean(vals))))
cd=pd.DataFrame(crows)
fig,ax=plt.subplots(figsize=(7.4,4.5)); corrs=["non_corr","corr_25","corr_50"]; cl=["0% (clean)","25%","50%"]
x=np.arange(len(corrs)); w=0.16
for i,m in enumerate(["thr_tuned","LR","RF","SVM","XGB"]):
    y=[cd[(cd["corr"]==c)&(cd.model==m)].f1.values[0] for c in corrs]
    ax.bar(x+(i-2)*w,y,w,color=C[m],label=LAB[m])
ax.set_xticks(x); ax.set_xticklabels(cl); ax.set_xlabel("Corruption level of Bob's records"); ax.set_ylabel("F1 score")
ax.set_ylim(0,1.08); ax.set_title("Linkage quality vs data corruption (ε = 7, 500 records)",fontweight="bold")
ax.legend(frameon=False,fontsize=8.5,ncol=3); style(ax); fig.tight_layout(); fig.savefig("figs/d4_corruption.png",bbox_inches="tight"); plt.close(fig)

# efficiency: mean secs per model at 500
eff=df[(df["size"]==500)&(df.model.isin(list(MODELS)))].groupby("model").secs.mean().round(2).to_dict()
data["eff_500"]=eff
data["corr"]={c:{m:round(cd[(cd["corr"]==c)&(cd.model==m)].f1.values[0],3) for m in ["thr_tuned","LR","RF","SVM","XGB"]} for c in corrs}
open("summary_d4.json","w").write(json.dumps(data,indent=0))
print("DONE %.0fs"%(time.time()-t0)); print(json.dumps(data["eff_500"])); print(json.dumps({str(k):v for k,v in data.items() if k in [(500,"7"),(500,"3"),(500,"1")]}))
