"""Three static, report-ready 3D companions to Task 1 Figs. 1--3.

Reads the completed Task 1 OMC coordinates and empirical marker reference;
does not modify or re-run the Task 1 preprocessing.  The synthetic gap at
297.63--297.72 s is reconstructed here only for a 3D demonstration.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "inputs" / "omc_refined_task1.csv"
REF = ROOT / "inputs" / "empirical_template_mm.csv"
FIG = ROOT / "figures"
BLUE = "#2c71b8"
RED = "#d74d42"
GRAY = "#8b939d"
INK = "#243447"
MARKER = ["#177d8b", "#dc775a", "#8066ac", "#c69b33"]


def data():
    d = pd.read_csv(SRC)
    t = d["omc_time"].to_numpy(float)
    p = d[[f"marker_{m}_{a}" for m in range(1, 5) for a in "XYZ"]].to_numpy(float).reshape(-1, 4, 3)
    ref = np.loadtxt(REF, delimiter=",")
    assert p.shape == (60806, 4, 3) and ref.shape == (4, 3)
    assert np.isfinite(p).all() and np.isfinite(ref).all()
    return t, p, ref


def proper_kabsch(source, target):
    """Proper 3D rotation for row-vector correspondences, det=+1."""
    a = source - source.mean(0)
    b = target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    handedness = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1., 1., handedness])
    r = u @ correction @ vt
    return r


def view_style(ax, title, xlab="X (mm)", ylab="Y (mm)", zlab="Z (mm)"):
    ax.set_title(title, color=INK, fontsize=13.2, weight="bold", pad=14)
    ax.set_xlabel(xlab, labelpad=8)
    ax.set_ylabel(ylab, labelpad=8)
    ax.set_zlabel(zlab, labelpad=8)
    ax.view_init(elev=26, azim=-57)
    ax.set_proj_type("ortho")
    ax.grid(True, alpha=.22)
    ax.tick_params(labelsize=8, pad=1)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((.94, .97, .99, .18))
        axis.line.set_color("#abb8c2")


def equal_scale(ax, points, pad=.16, min_spans=(20, 20, 20)):
    pts = np.asarray(points).reshape(-1, 3)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    mid = (lo + hi) / 2
    half = np.maximum((hi - lo) / 2 * (1 + pad), np.array(min_spans) / 2)
    for fun, m, h in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid, half):
        fun(m-h, m+h)
    # Equal physical scaling in X/Y/Z despite unequal spans.
    ax.set_box_aspect(half * 2, zoom=.85)


def fig1_geometry(t, p, ref):
    pairs = list(itertools.combinations(range(4), 2))
    lengths = np.column_stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in pairs])
    med = np.median(lengths, axis=0)
    absdiff = np.abs(lengths - med)
    typical = int(np.argmin(np.sum(absdiff**2, axis=1)))
    anomaly = int(np.argmax(np.sum(absdiff**2, axis=1)))

    fig = plt.figure(figsize=(13.8, 6.8), facecolor="white")
    fig.suptitle("Fig. 1 (3D)  Geometry of the six marker pairs", fontsize=17.5,
                 weight="bold", color=INK, y=.985)
    fig.text(.5, .922,
             "Representative frame near the pairwise medians vs. frame of maximum summed distance deviation",
             ha="center", fontsize=10.4, color="#586b7b")
    for k, idx in enumerate([typical, anomaly]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        pose = p[idx]
        # Visualization-only orientation normalization, using saved Task 1 reference.
        r = proper_kabsch(pose, ref)
        aligned = (pose - pose.mean(0)) @ r
        dev = lengths[idx] - med
        for j, (a, b) in enumerate(pairs):
            is_extreme = abs(dev[j]) > 5
            color = RED if k == 1 and is_extreme else "#a7b4bd"
            ax.plot(aligned[[a,b],0],aligned[[a,b],1],aligned[[a,b],2],
                    color=color, lw=2.6 if is_extreme and k == 1 else 1.45,
                    alpha=1 if is_extreme and k == 1 else .75)
        for m in range(4):
            ax.scatter(*aligned[m],s=130,color=MARKER[m],edgecolor="white",linewidth=1.5,depthshade=False)
            ax.text(*(aligned[m] + [3, 3, 2]),f"M{m+1}",color=MARKER[m],fontsize=11,weight="bold")
        view_style(ax, ("Typical" if k == 0 else "Largest deviation")+f"   ·   {t[idx]:.2f} s",
                   "Local X (mm)","Local Y (mm)","Local Z (mm)")
        equal_scale(ax,np.vstack((aligned,ref)),pad=.22,min_spans=(115,115,75))
        info=[f"{a+1}–{b+1}: {lengths[idx,j]:.1f}" for j,(a,b) in enumerate(pairs)]
        x=.255 if k==0 else .747
        fig.text(x,.102,"Pair distances (mm)",ha="center",fontsize=10,weight="bold",color=INK)
        fig.text(x,.070,"     ".join(info[:3]),ha="center",fontsize=8.6,color="#526476")
        fig.text(x,.048,"     ".join(info[3:]),ha="center",fontsize=8.6,color="#526476")
    fig.text(.5,.006,"Red links are more than 5 mm from the corresponding pair median in this visualization; axes were orientation-normalized for comparison.",
             ha="center",fontsize=8.6,color="#6c7883")
    fig.subplots_adjust(left=.015,right=.985,bottom=.15,top=.86,wspace=.03)
    fig.savefig(FIG/"Fig1_Marker_Pair_Geometry_3D.png",dpi=230,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return {"typical_time_s":float(t[typical]),"anomalous_time_s":float(t[anomaly]),
            "pair_medians_mm":{f"{a+1}-{b+1}":float(x) for (a,b),x in zip(pairs,med)}}


def single_marker_reconstruction(t,p,ref,gap):
    src=ref[1:]
    mean_src=src.mean(axis=0)
    result=[]
    for visible in p[gap,1:]:
        r=proper_kabsch(src,visible)
        result.append((ref[0]-mean_src)@r + visible.mean(axis=0))
    return np.asarray(result)


def all_marker_spline(t,p,gap):
    # Local cubic splines fitted to available coordinates outside the synthetic gap.
    context=(t>=297.3)&(t<=298.1)&(~gap)
    vals=p[context].reshape(-1,12)
    return CubicSpline(t[context],vals,axis=0)(t[gap]).reshape(-1,4,3)


def legend_handles():
    return [Line2D([0],[0],color=BLUE,lw=2.5,label="Available coordinates"),
            Line2D([0],[0],color=GRAY,lw=1.8,ls="--",label="Original / withheld truth"),
            Line2D([0],[0],color=RED,marker="o",lw=1.5,ms=5,label="Reconstructed gap")]


def fig2_one_marker(t,p,gap,pred):
    sel=(t>=297.30)&(t<=298.10)
    before=sel & (t<297.63-1e-7)
    after=sel & (t>297.72+1e-7)
    true=p[sel,0]
    fig=plt.figure(figsize=(14,6.75),facecolor="white")
    gt=p[gap,0]
    near=(t>=297.56)&(t<=297.79)
    near_before=near & (t<297.63-1e-7)
    near_after=near & (t>297.72+1e-7)
    for k in [0,1]:
        ax=fig.add_subplot(1,2,k+1,projection="3d")
        truth=true if k==0 else p[near,0]
        ax.plot(truth[:,0],truth[:,1],truth[:,2],color=GRAY,ls="--",lw=1.6,alpha=.82)
        for observed in ([before,after] if k==0 else [near_before,near_after]):
            obs=p[observed,0]
            ax.plot(obs[:,0],obs[:,1],obs[:,2],color=BLUE,lw=2.7)
        ax.plot(pred[:,0],pred[:,1],pred[:,2],color=RED,ls="--",lw=1.35,alpha=.9)
        ax.scatter(pred[:,0],pred[:,1],pred[:,2],s=48,color=RED,
                   edgecolors="white",linewidth=.65,depthshade=False)
        ax.scatter(gt[:,0],gt[:,1],gt[:,2],s=21,facecolors="none",
                   edgecolors=GRAY,linewidths=1,depthshade=False)
        for r,q in zip(pred[::2],gt[::2]):
            ax.plot([r[0],q[0]],[r[1],q[1]],[r[2],q[2]],color="#cb7770",lw=1.1,alpha=.8)
        view_style(ax,("Full interval: 297.30–298.10 s" if k==0 else "Gap magnified: 297.56–297.79 s"),
                   "X (mm)","Y (mm)","Z (mm)")
        equal_scale(ax,np.vstack((truth,pred)),pad=.21 if k==0 else .40,
                    min_spans=(20,20,20))
        ax.view_init(elev=26,azim=-53 if k==1 else -62)
    fig.suptitle("Fig. 2 (3D)  Rigid-body reconstruction of Marker 1",y=.982,
                 weight="bold",fontsize=17.5,color=INK)
    fig.legend(handles=legend_handles(),loc="lower center",bbox_to_anchor=(.5,.083),ncol=3,
               frameon=False,fontsize=10)
    fig.text(.5,.039,"Removed time: 297.63–297.72 s (10 samples). Lines joining red and gray points illustrate reconstruction error.",
             ha="center",fontsize=9.5,color="#566778")
    fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.88,wspace=.025)
    fig.savefig(FIG/"Fig2_One_Marker_Rigid_Reconstruction_3D.png",dpi=230,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return float(np.sqrt(np.mean(np.sum((pred-gt)**2,axis=1))))


def fig3_four_markers(t,p,gap,pred):
    sel=(t>=297.30)&(t<=298.10)
    before=sel & (t<297.63-1e-7)
    after=sel & (t>297.72+1e-7)
    fig=plt.figure(figsize=(11.3,7.7),facecolor="white")
    ax=fig.add_subplot(111,projection="3d")
    for j in range(4):
        original=p[sel,j]
        ax.plot(original[:,0],original[:,1],original[:,2],color=GRAY,ls="--",lw=1.1,alpha=.65)
        for observed in [before,after]:
            q=p[observed,j]
            ax.plot(q[:,0],q[:,1],q[:,2],color=BLUE,lw=2.1,alpha=.9)
        rec=pred[:,j]
        ax.plot(rec[:,0],rec[:,1],rec[:,2],color=RED,lw=.9,alpha=.72)
        ax.scatter(rec[:,0],rec[:,1],rec[:,2],color=RED,marker="x",s=30,linewidths=1.25,depthshade=False)
        anchor=p[before,j][0]
        ax.text(*(anchor + [1.3,1.3,1.3]),f"M{j+1}",color=MARKER[j],fontsize=11.6,weight="bold")
    view_style(ax,"All four markers · cubic-spline reconstruction")
    equal_scale(ax,p[sel].reshape(-1,3),pad=.23)
    ax.view_init(elev=25,azim=-58)
    fig.suptitle("Fig. 3 (3D)  Simultaneous synthetic gap in all markers",y=.980,
                 weight="bold",fontsize=17.0,color=INK)
    legend=legend_handles()
    legend[2]=Line2D([0],[0],color=RED,marker="x",lw=1.4,ms=7,label="Reconstructed gap")
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.065),ncol=3,
               frameon=False,fontsize=10)
    fig.text(.5,.032,"Removed time: 297.63–297.72 s (10 samples). Red crosses show interpolated X, Y and Z coordinates.",
             ha="center",fontsize=9.5,color="#566778")
    fig.subplots_adjust(left=.02,right=.98,bottom=.105,top=.87)
    fig.savefig(FIG/"Fig3_Four_Marker_Spline_Reconstruction_3D.png",dpi=230,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return float(np.sqrt(np.mean(np.sum((pred-p[gap])**2,axis=2))))


def main():
    FIG.mkdir(exist_ok=True)
    t,p,ref=data()
    gap=(t>=297.63-1e-8)&(t<=297.72+1e-8)
    assert int(gap.sum())==10
    f1=fig1_geometry(t,p,ref)
    rigid=single_marker_reconstruction(t,p,ref,gap)
    cubic=all_marker_spline(t,p,gap)
    err1=fig2_one_marker(t,p,gap,rigid)
    err4=fig3_four_markers(t,p,gap,cubic)
    summary={"scope":"Standalone 3D visualization of Task 1 section 3.1; Task 1 originals unchanged",
             "synthetic_gap_s":[297.63,297.72],"gap_frames":int(gap.sum()),"figure1":f1,
             "figure2_independent_kabsch_vector_RMSE_mm":err1,
             "figure3_independent_cubic_spline_vector_RMSE_mm":err4,
             "important_note":"These computations are illustrations from the stored Task 1 CSV, not a reproduction of the unavailable original gap-reconstruction script. Published report Table 1 values should not be silently replaced or claimed reproduced."}
    (FIG/"figure_recomputation_notes.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"synthetic_gap_frames":int(gap.sum()),"rigid_vector_rmse_mm":err1,"cubic_vector_rmse_mm":err4},indent=2))


if __name__=="__main__":
    main()
