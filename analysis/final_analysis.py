import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE=os.path.dirname(os.path.abspath(__file__))
from load import load
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman","DejaVu Serif"],
    "font.size":8,"axes.titlesize":8,"axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.4})
DS=["fraud","covtype","protein_homo"]; DSN={"fraud":"Credit-card fraud","covtype":"Covertype","protein_homo":"protein_homo"}
MD=["lr","rf","hgb"]; MN={"lr":"LR","rf":"RF","hgb":"HGB"}; COL={"lr":"#1f77b4","rf":"#d62728","hgb":"#2ca02c"}
rng=np.random.default_rng(0)
def ci(x,n=2000):
    x=np.asarray(x,float); m=rng.choice(x,(n,len(x))).mean(1); return x.mean(),np.quantile(m,.025),np.quantile(m,.975)
def add(d):
    for c in (10,50,200):
        d[f"rcr_f1_{c}"]=(d[f"cost_c{c}_default"]-d[f"cost_c{c}_tuned"])/d[f"cost_c{c}_default"]
        d[f"rcr_ct_{c}"]=(d[f"cost_c{c}_default"]-d[f"cost_c{c}_ctuned"])/d[f"cost_c{c}_default"]
    return d
R={k:add(load(k)) for k in ["A","B","C","D"]}
out={}; key={}

# ---------- Fig 1: ratio (C vs D) ----------
fig,axs=plt.subplots(1,3,figsize=(7.0,2.35),sharey=True)
for ax,ds in zip(axs,DS):
    for m in MD:
        for k,ls,lab in (("C","-","fixed positives"),("D","--","fixed size")):
            g=R[k][(R[k].dataset==ds)&(R[k].model==m)].groupby("x").rcr_ct_50.mean()
            ax.plot(g.index,g.values,ls=ls,marker="o",ms=2.5,lw=1.1,color=COL[m],label=f"{MN[m]}, {lab}")
    ax.set_xscale("log"); xs=sorted(R["C"][R["C"].dataset==ds].x.unique()); ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mt.ScalarFormatter()); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_title(DSN[ds]); ax.set_xlabel("Negatives per positive (log scale)")
axs[0].set_ylabel("Relative cost reduction"); axs[0].set_ylim(-0.05,1.0)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,ncol=6,loc="lower center",frameon=False,bbox_to_anchor=(0.5,-0.02),columnspacing=1.0,handlelength=1.8)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig(os.path.join(HERE,"figs","fig1_ratio.png"),dpi=300); plt.close(fig)

# ---------- Fig 2: training positives (A) ----------
fig,axs=plt.subplots(1,3,figsize=(7.0,2.3),sharey=True)
for ax,ds in zip(axs,DS):
    for m in MD:
        g=R["A"][(R["A"].dataset==ds)&(R["A"].model==m)]
        xs=sorted(g.x.unique()); st=[ci(g[g.x==x].benefit_f1) for x in xs]
        mu=[s[0] for s in st]; lo=[s[1] for s in st]; hi=[s[2] for s in st]
        ax.plot(xs,mu,marker="o",ms=2.5,lw=1.1,color=COL[m],label=MN[m]); ax.fill_between(xs,lo,hi,color=COL[m],alpha=0.18,lw=0)
    ax.axhline(0,color="k",lw=0.7); ax.set_xscale("log"); ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mt.ScalarFormatter()); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_title(DSN[ds]); ax.set_xlabel("Training positives (log scale)")
axs[0].set_ylabel("F1 gain from tuned threshold"); axs[0].legend(frameon=False,loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE,"figs","fig2_trainpos.png"),dpi=300); plt.close(fig)

# ---------- Fig 3: validation positives (B) ----------
fig,axs=plt.subplots(2,3,figsize=(7.0,3.9),sharex=True)
for j,ds in enumerate(DS):
    for m in MD:
        g=R["B"][(R["B"].dataset==ds)&(R["B"].model==m)]
        xs=sorted(g.x.unique()); st=[ci(g[g.x==x].benefit_f1) for x in xs]
        axs[0,j].plot(xs,[s[0] for s in st],marker="o",ms=2.5,lw=1.1,color=COL[m],label=MN[m])
        axs[0,j].fill_between(xs,[s[1] for s in st],[s[2] for s in st],color=COL[m],alpha=0.18,lw=0)
        axs[1,j].plot(xs,[g[g.x==x].chosen_threshold.std() for x in xs],marker="o",ms=2.5,lw=1.1,color=COL[m])
    axs[0,j].axhline(0,color="k",lw=0.7); axs[0,j].set_title(DSN[ds])
    axs[1,j].set_xscale("log"); axs[1,j].set_xticks(xs)
    axs[1,j].xaxis.set_major_formatter(mt.ScalarFormatter()); axs[1,j].xaxis.set_minor_formatter(mt.NullFormatter())
    axs[1,j].set_xlabel("Validation positives (log scale)")
axs[0,0].set_ylabel("F1 gain from tuned threshold"); axs[1,0].set_ylabel("SD of chosen threshold"); axs[0,0].legend(frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE,"figs","fig3_valpos.png"),dpi=300); plt.close(fig)

# ---------- Table III: headline ----------
rows=[]
def span(k,ds,m):
    g=R[k][(R[k].dataset==ds)&(R[k].model==m)]; xs=sorted(g.x.unique()); lo,hi=xs[0],xs[-1]
    a=g[g.x==lo].set_index("seed").rcr_ct_50; b=g[g.x==hi].set_index("seed").rcr_ct_50
    d=(b-a).dropna(); return lo,hi,ci(a)[0],ci(b)[0],ci(d), int(g[g.x==hi].n_train_pos.iloc[0])
for ds in DS:
    for m in MD:
        lo,hi,aD,bD,dD,ntr=span("D",ds,m); lo2,hi2,aC,bC,dC,_=span("C",ds,m)
        rows.append(dict(dataset=DSN[ds],model=MN[m],range=f"1:{lo}-1:{hi}",ntr_hi=ntr,
            D_lo=aD,D_hi=bD,D_chg=dD[0],D_ci=(dD[1],dD[2]),C_lo=aC,C_hi=bC,C_chg=dC[0],C_ci=(dC[1],dC[2])))
out["table3"]=rows
for r in rows: print(r["dataset"],r["model"],r["range"],r["ntr_hi"],f'D {r["D_lo"]:.3f}->{r["D_hi"]:.3f} chg {r["D_chg"]:+.3f} [{r["D_ci"][0]:+.3f},{r["D_ci"][1]:+.3f}] | C {r["C_lo"]:.3f}->{r["C_hi"]:.3f} chg {r["C_chg"]:+.3f} [{r["C_ci"][0]:+.3f},{r["C_ci"][1]:+.3f}]')

# ---------- strategies ----------
S=pd.concat([load(k).assign(exp=k) for k in ("S_A","S_C","S_D")],ignore_index=True)
S["rcr_ct_50"]=(S.cost_c50_default-S.cost_c50_ctuned)/S.cost_c50_default
kk=["dataset","exp","model","x","seed"]
b=S[S.strategy=="none"].set_index(kk)
S=S.join(b[["f1_tuned","cost_c50_ctuned","cost_c50_default","ece"]].add_prefix("n_"),on=kk)
S["d_f1_def"]=S.f1_default-S.n_f1_tuned; S["d_f1_tun"]=S.f1_tuned-S.n_f1_tuned
S["rel_def"]=(S.cost_c50_default-S.n_cost_c50_ctuned)/S.n_cost_c50_ctuned
S["rel_ct"]=(S.cost_c50_ctuned-S.n_cost_c50_ctuned)/S.n_cost_c50_ctuned
ST=["cw","ros","rus","smote"]; STN={"none":"Plain","cw":"Class weights","ros":"Oversampling","rus":"Undersampling","smote":"SMOTE"}
SHORT={"none":"Plain","cw":"Class\nweights","ros":"Over-\nsampling","rus":"Under-\nsampling","smote":"SMOTE"}
t4=[]
for st in ST:
    for m in MD:
        r={"strategy":STN[st],"model":MN[m]}
        for ds in DS:
            g=S[(S.dataset==ds)&(S.model==m)&(S.strategy==st)].groupby("seed").rel_ct.mean()
            mu,lo,hi=ci(g); r[ds]=(mu,lo,hi)
        t4.append(r)
out["table4"]=t4
for r in t4: print(r["strategy"],r["model"],*[f'{r[d][0]:+.3f}{"*" if r[d][1]>0 or r[d][2]<0 else ""}' for d in DS])
pool={}
for st in ["none"]+ST:
    for m in MD:
        g=S[(S.model==m)&(S.strategy==st)]
        pool[(st,m)]=dict(ece=g.groupby("seed").ece.mean().mean(),f1d=g.f1_default.mean(),f1t=g.f1_tuned.mean(),
                          d_def=ci(g.groupby("seed").d_f1_def.mean()) if st!="none" else None,
                          d_tun=ci(g.groupby("seed").d_f1_tun.mean()) if st!="none" else None,
                          rel_def=ci(g.groupby("seed").rel_def.mean()) if st!="none" else None)
out["table5"]=[dict(strategy=STN[st],model=MN[m],**{k:v for k,v in pool[(st,m)].items()}) for st in ["none"]+ST for m in MD]
for r in out["table5"]: print(r["strategy"],r["model"],f'ECE {r["ece"]:.4f} F1@.5 {r["f1d"]:.3f} F1tuned {r["f1t"]:.3f}', r["d_def"] and f'd_def {r["d_def"][0]:+.3f}', r["d_tun"] and f'd_tun {r["d_tun"][0]:+.3f}', r["rel_def"] and f'cost@.5 vs plain-ct {r["rel_def"][0]:+.2f}')

# ---------- Fig 4: strategies ----------
fig,axs=plt.subplots(1,2,figsize=(7.0,2.5))
w=0.26; xs=np.arange(len(["none"]+ST))
for i,m in enumerate(MD):
    axs[0].bar(xs+(i-1)*w,[pool[(s,m)]["ece"] for s in ["none"]+ST],w,color=COL[m],label=MN[m])
axs[0].set_xticks(xs); axs[0].set_xticklabels([SHORT[s] for s in ["none"]+ST]); axs[0].set_ylabel("Mean ECE (threshold 0.5)")
axs[0].legend(frameon=False)
for i,m in enumerate(MD):
    axs[1].bar(np.arange(len(ST))+(i-1)*w,[pool[(s,m)]["rel_def"][0] for s in ST],w,color=COL[m])
axs[1].axhline(0,color="k",lw=0.7); axs[1].set_xticks(np.arange(len(ST))); axs[1].set_xticklabels([SHORT[s] for s in ST])
axs[1].set_ylabel("Extra cost at threshold 0.5 vs plain\nmodel with cost-optimal threshold")
fig.tight_layout(); fig.savefig(os.path.join(HERE,"figs","fig4_strat.png"),dpi=300); plt.close(fig)

# ---------- Table II-ish numbers and key text numbers ----------
# validation-positives effect
print("\nB: F1 gain @5 vs @80 (mean) and CI at 5")
for ds in DS:
    for m in MD:
        g=R["B"][(R["B"].dataset==ds)&(R["B"].model==m)]
        a=ci(g[g.x==5].benefit_f1); b=ci(g[g.x==80].benefit_f1)
        rc=ci(g[g.x==5].rcr_ct_50); rd=ci(g[g.x==80].rcr_ct_50)
        print(ds,m,f"F1gain5 {a[0]:+.3f}[{a[1]:+.3f},{a[2]:+.3f}] F1gain80 {b[0]:+.3f} | RCRct5 {rc[0]:+.3f}[{rc[1]:+.3f},{rc[2]:+.3f}] RCRct80 {rd[0]:+.3f}")
print("\nA: F1 gain with 10 vs 160 training positives")
for ds in DS:
    for m in MD:
        g=R["A"][(R["A"].dataset==ds)&(R["A"].model==m)]
        a=ci(g[g.x==10].benefit_f1); b=ci(g[g.x==160].benefit_f1)
        print(ds,m,f"{a[0]:+.3f}[{a[1]:+.3f},{a[2]:+.3f}] -> {b[0]:+.3f}[{b[1]:+.3f},{b[2]:+.3f}]")
print("\nOverall RCR with F1-thr vs cost-thr, c=50, pooled over A,C,D,B")
al=pd.concat(R.values())
print(al.groupby("dataset")[["rcr_f1_50","rcr_ct_50"]].mean().round(3))
print((al.rcr_ct_50>al.rcr_f1_50).mean())
json.dump(out,open(os.path.join(HERE,"tables.json"),"w"),default=float)
