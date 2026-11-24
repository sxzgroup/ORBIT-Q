"""Rebuild this draft's five charts from its frozen JSON data. Requires matplotlib and numpy."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
def solver_label(v):
    return {"sol":"6.2 Sol high","astra":"6.2 Astra high","terra":"6.1 Terra high"}[v]
OLD="#0072B2 #D55E00 #009E73 #CC79A7 #B2182B #222222 #56B4E9 #8C564B #7B2CBF".split()
FAMILY={"sol":"#004B85","astra":"#6932A0","terra":"#00784F"}
# Exact names in the two frozen input tables; do not filter whole model families.
RETIRED={"5.6 Sol high":"Sol","5.6 Sol ultra":"Sol ultra","5.6 Terra high":"Terra",
         "5.6 Luna high":"Luna","Grok 4.5 high":"Grok 4.5"}
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":9,
                    "font.family":"DejaVu Sans","svg.fonttype":"none","svg.hashsalt":"orbitq-62"})

def save(fig,dest,name):
    figs=dest/"figures"
    figs.mkdir(exist_ok=True)
    for fmt in ("png","svg","pdf"):
        opts={"svg":{"Date":None},"pdf":{"CreationDate":None,"ModDate":None}}
        fig.savefig(figs/f"{name}.{fmt}",dpi=200,metadata=opts.get(fmt,{}),facecolor="white")
        if fmt=="svg":
            path=figs/f"{name}.{fmt}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)

def outcomes(runs,out):
    x=np.arange(1,13)
    fig,ax=plt.subplots(figsize=(9.4,0.38*len(runs)+1.9),layout="constrained")
    m=np.array([[n in r["passes"] for n in x] for r in runs])
    ax.imshow(m,cmap=ListedColormap(["#F7F7F7","#D9ECE7"]),vmin=0,vmax=1,aspect="auto")
    for i,row in enumerate(m):
        for j,p in enumerate(row):
            ax.text(j,i,"P" if p else "F",ha="center",va="center",color="#175B48" if p else "#444444",weight="bold")
    ax.set(xticks=x-1,xticklabels=[f"{n:02d}" for n in x],yticks=range(len(runs)),
           yticklabels=[f"{r['label']}   {len(r['passes'])}/12" for r in runs],xlabel="Challenge",
           title="TensorCircuit task outcomes")
    ax.set_xticks(np.arange(-.5,12,1),minor=True); ax.set_yticks(np.arange(-.5,len(runs),1),minor=True)
    ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="both",length=0)
    save(fig,out,"task-outcomes")

def efficiency(ax,runs):
    filtered=[r for r in runs if len(r["passes"]) and r["total_cost_usd"] is not None]
    for i,r in enumerate(filtered,1):
        p=len(r["passes"]); total=r["total_cost_usd"]
        x=r["agent_wall_sec"]/p/60; y=total/p
        ax.scatter(x,y,s=total*25,color=r["color"],edgecolors="black",linewidths=.7,alpha=.84,zorder=3)
        ax.annotate(str(i),(x,y),xytext=(3,4),textcoords="offset points",fontsize=7,weight="bold",zorder=4,
                    bbox=dict(boxstyle="round,pad=.10",facecolor="white",edgecolor="none",alpha=.8))
    ax.set(xlabel="Solver minutes per pass",ylabel="Solver USD per pass",title="(c)  Resource efficiency")
    ax.set_xlim(0,max(74,max(r["agent_wall_sec"]/len(r["passes"])/60 for r in filtered)*1.12))
    ax.set_ylim(0,max(r["total_cost_usd"]/len(r["passes"]) for r in filtered)*1.18)
    ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
    return [Line2D([],[],linestyle="none",marker="o",markersize=6,color=r["color"],
             label=f"{i}  {r['label']}  ({len(r['passes'])}/12)") for i,r in enumerate(filtered,1)]

def resources(v,rows,runs,out):
    x=np.arange(1,13); fig,axes=plt.subplots(1,3,figsize=(13,5.45))
    fig.subplots_adjust(left=.055,right=.985,top=.82,bottom=.36,wspace=.34)
    wall=np.array([r["solver_wall_sec"]/60 for r in rows])
    axes[0].bar(x,wall,color="#009E73",edgecolor="black",lw=.5)
    axes[0].set(title=f"(a)  Solver time · {wall.sum():.2f} min",ylabel="Solver wall time (min)")
    inp=np.array([r["n_input_tokens"] for r in rows],dtype=float)
    cache=np.array([r["n_cache_tokens"] for r in rows],dtype=float)
    output=np.array([r["n_output_tokens"] for r in rows],dtype=float)
    bottom=np.zeros(12)
    for arr,color,lab in [(inp-cache,"#595959","Non-cache prompt"),(cache,"#56B4E9","Cache-read prompt"),(output,"#D55E00","Output")]:
        axes[1].bar(x,arr/1e6,bottom=bottom,color=color,edgecolor="black",lw=.4,label=lab); bottom+=arr/1e6
    axes[1].set(title=f"(b)  Solver tokens · {bottom.sum():.3f}M",ylabel="Tokens (million)")
    axes[1].legend(fontsize=7,frameon=False)
    for a in axes[:2]:
        a.set(xticks=x,xticklabels=[f"{n:02d}" for n in x],xlabel="Challenge")
        a.tick_params(axis="x",labelsize=7); a.grid(axis="y",alpha=.18); a.set_axisbelow(True)
    handles=efficiency(axes[2],runs); axes[2].set_box_aspect(1)
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.052),ncol=4,frameon=False,fontsize=8)
    fig.suptitle(f"GPT-{solver_label(v)} · agent-side resource use",x=.055,ha="left",weight="bold",fontsize=15)
    fig.text(.055,.032,"Marker area is proportional to total solver cost. Verifier resources excluded.",fontsize=8,color="#555555")
    save(fig,out,"agent-resources")
    fig=plt.figure(figsize=(6.5,6.5))
    ax=fig.add_axes([.22,.35,.59,.59]); ax.set_box_aspect(1); handles=efficiency(ax,runs)
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=8)
    fig.text(.5,.026,"Marker area is proportional to total solver cost.",ha="center",fontsize=8)
    save(fig,out,"resource-efficiency-square")

def paired_fig(v,out,pairs):
    fig,(a,b)=plt.subplots(1,2,figsize=(11.8,4.95),layout="constrained",
                          gridspec_kw={"width_ratios":[1.05,1.25]})
    x=np.arange(1,13); ex=np.array([r["expert_reference_sec"] if r["expert_reference_basis"]=="measured" and r["expert_reference_sec"] else np.nan for r in pairs]); model=np.array([r["model_sec"] or np.nan for r in pairs])
    speed=ex/model
    a.bar(x-.18,ex,.36,color="#808080",edgecolor="#333333",lw=.5,label="Expert reference")
    estimated=pairs[7]["expert_reference_basis"]=="estimated"
    proxy=bool(estimated and np.isfinite(model[7]))
    if estimated:
        a.bar(8-.18,pairs[7]["expert_reference_sec"],.36,color="white",edgecolor="#646464",hatch="////",lw=.65,label="Task 08: 50 s proxy")
    a.bar(x+.18,model,.36,color=FAMILY[v],edgecolor="#333333",lw=.5,label=solver_label(v))
    a.set_yscale("log"); a.set(ylim=(min(.8,np.nanmin([ex,model])*.8),np.nanmax([ex,model])*1.65),xticks=x,xticklabels=[f"{i:02d}" for i in x],
                               xlabel="Challenge",ylabel="End-to-end runtime (s, log scale)",title="(a)  Artifact execution time")
    a.legend(frameon=False,fontsize=8)
    b.set_xscale("log"); b.barh(x,speed,height=.62,color=FAMILY[v],edgecolor="#333333",lw=.6)
    if proxy:
        est=pairs[7]["expert_reference_sec"]/model[7]
        b.barh(8,est,height=.62,color="white",edgecolor=FAMILY[v],hatch="////",lw=.8)
    b.axvline(1,color="#777777",ls="--",lw=1)
    b.set(xlim=(min(.1,np.nanmin(speed)*.7),max(3,np.nanmax(speed)*2.2,est*2.2 if proxy else 0)),yticks=x,yticklabels=[f"Task {i:02d}" for i in x],
          xlabel="Speedup (expert / artifact, ×, log scale)",title=f"(b)  Speedup · {np.isfinite(speed).sum()+int(proxy)} valid ratios")
    b.invert_yaxis()
    for j,z in enumerate(speed):
        if proxy and j==7:
            b.text(est*1.07,8,f"{est:.2f}× (proxy)",va="center",fontsize=8);continue
        t=f"{z:.2f}×" if np.isfinite(z) else ("No valid submission" if pairs[j]["model_sec"] is None and j==7 and v=="terra" else "No measured pair")
        b.text(z*1.07 if np.isfinite(z) else 1.08,j+1,t,va="center",fontsize=8)
    for ax,axis in [(a,"y"),(b,"x")]:ax.grid(axis=axis,alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f"{solver_label(v)} · artifact runtime relative to the expert reference",fontweight="bold")
    save(fig,out,"artifact-vs-expert")

def slowdowns(dest,data):
    fig,ax=plt.subplots(figsize=(8.3,5),layout="constrained")
    ax.axhline(1,ls="--",color="#888888",lw=.8); ax.axvline(0,ls="--",color="#888888",lw=.8)
    ax.scatter(0,1,marker="D",s=85,facecolors="none",edgecolors="#555555",label="Expert TC reference",zorder=3)
    for i,(label,x,y) in enumerate(data["historical"]):
        if label in RETIRED.values(): continue
        ax.scatter(x,y,s=36,color=("#7B2CBF" if label=="Astra" else plt.cm.tab20(i/19)),edgecolors="#555555",lw=.6,
                   label=("6 Astra" if label=="Astra" else label),zorder=3)
    for row in data["new"]:
        if row["ratio"] is not None:
            ax.scatter(row["failure_pct"],row["ratio"],marker=row["marker"],s=108,color=FAMILY[row["variant"]],edgecolors="black",lw=.7,label=f"{row['label']} ({row['pairs']} pairs)",zorder=4)
    ax.set(xlim=(-8,max(68,ax.get_xlim()[1])),ylim=(-.45,max(8.3,ax.get_ylim()[1])),xlabel="Failure rate (%)",ylabel="Artifact time / expert time (geometric mean)",title="Validity and artifact efficiency")
    zoom=ax.inset_axes([.08,.64,.30,.25])
    zoom.set_facecolor("#FAFAFA"); zoom.axhline(1,ls=":",color="#AAAAAA",lw=.7)
    close=[("Expert",1,"D","#888888"),("6 Astra",next(y for l,x,y in data["historical"] if l=="Astra"),"o","#7B2CBF")]
    close.extend((r["label"].removesuffix(" high"),r["ratio"],r["marker"],FAMILY[r["variant"]]) for r in data["new"] if r["failure_pct"]==0)
    for name,y,m,c in close:
        zoom.scatter(0,y,s=24,marker=m,facecolors="none" if name=="Expert" else c,edgecolors=c,lw=.8)
        zoom.text(.10,y,name,va="center",fontsize=6.9,color="#343434")
    zoom.set(xlim=(-.17,1.30),ylim=(.02,1.14),xticks=[],yticks=[.2,.6,1])
    zoom.tick_params(labelsize=6,length=2);zoom.set_title("Zero-failure configurations",fontsize=7.2,pad=2)
    ax.legend(loc="center left",bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
    save(fig,dest,"artifact-efficiency")


def main():
    out=Path(__file__).resolve().parent
    read=lambda name: json.loads((out/name).read_text())
    v=read("execution-protocol.json")["solver"].split("-")[-1]
    rows=read("trials.json")
    runs=[r for r in read("resources-comparison.json") if r["label"] not in RETIRED]
    outcomes(runs,out)
    resources(v,rows,runs,out)
    paired_fig(v,out,read("published-runtimes.json"))
    slowdowns(out,read("runtime-comparison.json"))

if __name__=="__main__": main()
