"""Create a static and standalone interactive 3-D view of existing Task 1 outputs.

This is visualization only: neither marker reconstruction nor reference fitting
is recomputed.  The original 100-Hz Task 1 CSV is used unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
OUT = HERE if (HERE / "inputs" / "omc_refined_task1.csv").is_file() else HERE / "Task1_3D_Visualization"
INPUT = OUT / "inputs"
OUTPUT = OUT / "visuals"
MARKER_COLORS = ["#1b8d9b", "#e37657", "#8665bb", "#e1aa42"]
DATA = INPUT / "omc_refined_task1.csv"
TEMPLATE = INPUT / "empirical_template_mm.csv"


def load_task1():
    data = pd.read_csv(DATA)
    expected = ["omc_time"] + [f"marker_{m}_{axis}" for m in range(1, 5) for axis in "XYZ"]
    if list(data.columns) != expected:
        raise ValueError("Unexpected Task 1 column order")
    t = data["omc_time"].to_numpy(float)
    marks = data.iloc[:, 1:].to_numpy(float).reshape(-1, 4, 3)
    ref = np.loadtxt(TEMPLATE, delimiter=",")
    if marks.shape != (60806, 4, 3) or ref.shape != (4, 3):
        raise ValueError("Saved Task 1 dimensions differ from the expected data")
    if not (np.isfinite(marks).all() and np.isfinite(ref).all()):
        raise ValueError("Non-finite coordinates in Task 1 outputs")
    return t, marks, ref


def make_png(t, marks, ref):
    c = marks.mean(axis=1)
    fig = plt.figure(figsize=(16.6, 8.6), facecolor="white")
    fig.text(.055, .946, "TASK 1  |  3D OPTICAL MARKER RESULTS", fontsize=22,
             fontweight="bold", color="#183142", family="DejaVu Sans")
    fig.text(.055, .902,
             "Refined 100 Hz OMC positions  /  ENU global frame  /  608.05 s  /  original Task 1 data unchanged",
             fontsize=10.7, color="#536678")

    a = fig.add_axes([.042, .185, .572, .57], projection="3d")
    a.view_init(elev=27, azim=-61)
    stride = 22
    for j, color in enumerate(MARKER_COLORS):
        a.plot(marks[::stride, j, 0] / 1000, marks[::stride, j, 1] / 1000,
               marks[::stride, j, 2] / 1000, color=color, alpha=.18, lw=.60)
    a.plot(c[::stride, 0] / 1000, c[::stride, 1] / 1000, c[::stride, 2] / 1000,
           color="#193d54", alpha=.84, lw=1.34)
    a.scatter(*list(c[0] / 1000), color="#39a676", marker="o", s=90,
              edgecolors="white", linewidths=1.4, depthshade=False)
    a.scatter(*list(c[-1] / 1000), color="#db4d45", marker="X", s=100,
              edgecolors="white", linewidths=1.1, depthshade=False)
    fig.text(.08, .810, "A   Full 3D marker trajectories", fontsize=14,
             fontweight="bold", color="#193142")
    a.set_xlabel("East / X (m)", labelpad=12)
    a.set_ylabel("North / Y (m)", labelpad=12)
    a.set_zlabel("Up / Z (m)", labelpad=9)
    a.set_box_aspect((1.7, 5, .82), zoom=.87)
    a.tick_params(labelsize=9, pad=0)
    a.grid(True, alpha=.26)
    a.set_facecolor("#f8fbfd")

    b = fig.add_axes([.650, .24, .325, .54], projection="3d")
    b.view_init(elev=43, azim=-56)
    for j, color in enumerate(MARKER_COLORS):
        b.scatter(ref[j, 0], ref[j, 1], ref[j, 2], s=175, color=color,
                  edgecolors="white", linewidths=2, depthshade=False)
        b.text(ref[j, 0] + 4, ref[j, 1] + 1, ref[j, 2] + 2,
               f"M{j + 1}", fontsize=12, weight="bold", color=color)
    loop = ref[[0, 1, 2, 3, 0]]
    b.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#415f71", lw=2.5)
    b.scatter([0], [0], [0], color="#32495b", marker="+", s=80, linewidths=2)
    b.set_xlim(-62, 62)
    b.set_ylim(-62, 62)
    b.set_zlim(-62, 62)
    b.set_box_aspect((1, 1, 1), zoom=.99)
    b.set_xlabel("Local X (mm)", labelpad=3)
    b.set_ylabel("Local Y (mm)", labelpad=3)
    b.set_zlabel("Local Z (mm)", labelpad=3)
    b.tick_params(labelsize=7.5, pad=-2)
    b.set_title("B   Saved reference marker layout", fontsize=14,
                fontweight="bold", color="#193142", loc="left", pad=17)
    b.set_facecolor("#f8fbfd")

    fig.legend(handles=[Line2D([0], [0], color=MARKER_COLORS[i], lw=2.7,
                                label=f"Marker {i + 1}") for i in range(4)] +
               [Line2D([0], [0], color="#193d54", lw=2.7, label="Centroid"),
                Line2D([0], [0], marker="o", linestyle="", color="#39a676", label="Start"),
                Line2D([0], [0], marker="X", linestyle="", color="#db4d45", label="End")],
               loc="lower center", ncol=7, bbox_to_anchor=(.5, .105),
               frameon=False, fontsize=10)
    fig.text(.055, .042,
             "Panel A: trajectories in global ENU (m). Panel B: empirical Task 1 reference centered at its centroid (mm).",
             fontsize=9.4, color="#617080")
    fig.savefig(OUTPUT / "Task1_3D_Overview.png", dpi=185, facecolor="white")
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 1 | Interactive 3D Optical Marker Results</title>
<style>
:root{color-scheme:dark;--bg:#0c1826;--panel:#152739;--edge:#294155;--dim:#9ab0be;--text:#eef7fb;--accent:#71c9d5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,Arial}
header{max-width:1400px;margin:0 auto;padding:28px 22px 9px}small.k{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.2em}
h1{font-size:clamp(25px,3vw,39px);letter-spacing:-.04em;margin:5px 0 2px}p{margin:3px 0;color:var(--dim)}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:15px}.chip{padding:6px 13px;border:1px solid #28485c;border-radius:20px;background:#122638;color:#b5d5df;font-size:13px}
main{max-width:1400px;margin:auto;padding:18px 22px 40px}.dash{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}
.panel{background:linear-gradient(165deg,#172b3e,#102236);border:1px solid var(--edge);border-radius:17px;min-width:0;overflow:hidden}
.ph{padding:16px 18px 0;display:flex;justify-content:space-between;gap:12px;align-items:start}.ph b{font-size:16px}.ph span{font-size:12px;color:var(--dim)}
canvas{display:block;width:100%;height:470px;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}
.foot{padding:5px 16px 16px;color:#a7bdcb;font-size:12px}
.control{background:#152739;border:1px solid var(--edge);border-radius:17px;margin-top:16px;padding:17px 20px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.btn{border:1px solid #426071;border-radius:9px;background:#1a384b;color:white;padding:8px 15px;cursor:pointer}
.btn:hover{background:#235269}input[type=range]{accent-color:#74cbd8;min-width:170px;flex:1}select{border:1px solid #426071;border-radius:8px;background:#18384b;color:white;padding:8px}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:13px;font-size:13px;color:#c5d6df}.dot{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px}
.callout{margin-top:16px;color:#acc2d0;font-size:13px}.metric{font-variant-numeric:tabular-nums;font-weight:700;color:#e8fbff}
@media(max-width:980px){.dash{grid-template-columns:1fr}canvas{height:370px}.ph{padding-top:11px}}
</style></head>
<body><header><small class="k">MOBI311 · TASK 1</small><h1>3D Optical Marker Results</h1>
<p>Existing refined marker positions • right-handed OMC global East–North–Up frame</p>
<div class="stats"><span class="chip">60,806 frames</span><span class="chip">100 Hz</span><span class="chip">608.05 s</span><span class="chip">4 markers</span><span class="chip">Task 1 unchanged</span></div></header>
<main><div class="dash">
<section class="panel"><div class="ph"><div><b>Full movement · ENU global frame</b><div><span>3D centroid + marker trajectories (meters)</span></div></div><button class="btn" id="resetG">Reset view</button></div>
<canvas id="global" aria-label="Rotatable global 3D trajectories"></canvas><div class="foot">Drag to rotate · mouse wheel to zoom · green = start, red = end</div></section>
<section class="panel"><div class="ph"><div><b>Marker cluster at selected time</b><div><span>Centered at four-marker centroid (millimeters)</span></div></div><button class="btn" id="resetL">Reset view</button></div>
<canvas id="local" aria-label="Rotatable 3D marker geometry"></canvas><div class="foot">Measured coordinates relative to current centroid, in global-axis directions</div></section></div>
<section class="control"><div class="controls"><button class="btn" id="play">▶ Play</button>
<label for="time" class="metric" id="timeLabel">0.0 s</label>
<input id="time" type="range" min="0" max="608" value="0" step="0.5" aria-label="OMC time in seconds">
<select id="speed"><option value="5">5× playback</option><option value="15" selected>15× playback</option><option value="30">30× playback</option></select>
<label><input id="trails" type="checkbox" checked> Show marker trails</label></div>
<div class="legend"> <span><i class="dot" style="background:#1b8d9b"></i>Marker 1</span><span><i class="dot" style="background:#e37657"></i>Marker 2</span>
<span><i class="dot" style="background:#9a7cd8"></i>Marker 3</span><span><i class="dot" style="background:#e1aa42"></i>Marker 4</span>
<span><i class="dot" style="background:#ffffff"></i>Centroid</span></div>
<p class="callout">Preview samples are displayed every 0.5 seconds for responsiveness; the original 100-Hz CSV is not downsampled or modified. The fixed marker layout in the saved Task 1 reference is shown separately in the PNG overview.</p>
<p class="callout">Current inter-marker distances: <span class="metric" id="distance"></span></p>
</section></main>
<script>
'use strict';
const frames=__FRAME_DATA__;
const colors=['#1b8d9b','#e37657','#9a7cd8','#e1aa42'];
const canvasG=document.getElementById('global'), canvasL=document.getElementById('local');
const uiTime=document.getElementById('time');
const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
for(const row of frames)for(const p of row[1])for(let j=0;j<3;j++){bounds.min[j]=Math.min(bounds.min[j],p[j]);bounds.max[j]=Math.max(bounds.max[j],p[j]);}
const gc=bounds.min.map((v,i)=>(v+bounds.max[i])/2);
const viewG={yaw:-0.68,pitch:0.57,zoom:1.02},viewL={yaw:-0.73,pitch:0.65,zoom:1.0};
let at=0,playing=false,lastFrame=0,showTrails=true;
function lerpPoint(a,b,k){return a.map((v,j)=>v+(b[j]-v)*k);}
function snapshot(sec){let ix=Math.min(frames.length-2,Math.max(0,Math.floor(sec*2))),row=frames[ix],nxt=frames[ix+1];let k=(sec-row[0])/(nxt[0]-row[0]);k=Math.max(0,Math.min(1,k));return row[1].map((p,j)=>lerpPoint(p,nxt[1][j],k));}
function centroid(points){return [0,1,2].map(j=>points.reduce((a,p)=>a+p[j],0)/4);}
function setupCanvas(c){let d=window.devicePixelRatio||1;let w=c.clientWidth,h=c.clientHeight;c.width=Math.round(w*d);c.height=Math.round(h*d);let ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);return [ctx,w,h];}
function proj(p,ctr,view,w,h,scale){let x=p[0]-ctr[0],y=p[1]-ctr[1],z=p[2]-ctr[2];let c=Math.cos(view.yaw),s=Math.sin(view.yaw);let xx=c*x-s*y,yy=s*x+c*y;let cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);let hgt=sp*yy+cp*z;return [w*.50+xx*scale,h*.54-hgt*scale,cp*yy-sp*z];}
function line(ctx,pts,stroke,width=1,alpha=1){if(pts.length<2)return;ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0],pts[i][1]);ctx.stroke();ctx.restore();}
function disc(ctx,p,r,color,border='#dbeef5'){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle=border;ctx.lineWidth=1.4;ctx.stroke();}
function axes(ctx,view,w,h){let ox=w-63,oy=h-63,sc=30;let cos=Math.cos(view.yaw),sin=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);let v=[[cos,sp*sin],[ -sin,sp*cos],[0,cp]];let cs=['#ff8f80','#86d5ab','#8fb9f3'];let names=['X','Y','Z'];for(let j=0;j<3;j++){let xx=ox+v[j][0]*sc,yy=oy-v[j][1]*sc;line(ctx,[[ox,oy],[xx,yy]],cs[j],2.2);ctx.fillStyle=cs[j];ctx.font='bold 12px system-ui';ctx.fillText(names[j],xx+3,yy-2);}}
function paintGlobal(){let [ctx,w,h]=setupCanvas(canvasG);ctx.clearRect(0,0,w,h);let range=Math.max(...bounds.max.map((v,i)=>v-bounds.min[i]));let scale=Math.min(w,h)*.78/range*viewG.zoom;
let chosen=snapshot(at),cc=centroid(chosen),pr=(p)=>proj(p,gc,viewG,w,h,scale);
let minz=bounds.min[2]-.06;
for(let a=-3;a<=3;a++){let yy=gc[1]+a*.7;line(ctx,[pr([bounds.min[0]-.1,yy,minz]),pr([bounds.max[0]+.1,yy,minz])],'#294255',.75,.5);}
for(let a=-2;a<=2;a++){let xx=gc[0]+a*.3;line(ctx,[pr([xx,bounds.min[1]-.2,minz]),pr([xx,bounds.max[1]+.2,minz])],'#294255',.75,.5);}
let cpath=frames.map(row=>pr(centroid(row[1])));line(ctx,cpath,'#d6ebf0',1.6,.75);
if(showTrails){for(let j=0;j<4;j++)line(ctx,frames.map(row=>pr(row[1][j])),colors[j],.8,.39);}
let cbeg=pr(centroid(frames[0][1])),cend=pr(centroid(frames[frames.length-1][1]));disc(ctx,cbeg,5.2,'#42bd88');disc(ctx,cend,5.3,'#e76666');
const poly=[...chosen,chosen[0]].map(pr);line(ctx,poly,'#eff8fc',1.8,.83);
disc(ctx,pr(cc),5.7,'#eef9fc');for(let j=0;j<4;j++)disc(ctx,pr(chosen[j]),6.5,colors[j]);
axes(ctx,viewG,w,h);ctx.fillStyle='#c7d8e4';ctx.font='12px system-ui';ctx.fillText('ENU · meters',18,h-22);}
function paintLocal(){let [ctx,w,h]=setupCanvas(canvasL);ctx.clearRect(0,0,w,h);let current=snapshot(at),cc=centroid(current);let rel=current.map(p=>p.map((v,j)=>(v-cc[j])*1000));let b=78;let scale=Math.min(w,h)*.76/(2*b)*viewL.zoom;let pr=p=>proj(p,[0,0,0],viewL,w,h,scale);
let grid=[];for(let k=-60;k<=60;k+=20){line(ctx,[pr([-70,k,0]),pr([70,k,0])],'#244158',.8,.65);line(ctx,[pr([k,-70,0]),pr([k,70,0])],'#244158',.8,.65);}
let corners=[rel[0],rel[1],rel[2],rel[3],rel[0]].map(pr);line(ctx,corners,'#dce8ee',2.5,.9);
let p0=pr([0,0,0]);disc(ctx,p0,3.1,'#fff');
for(let j=0;j<4;j++){let p=pr(rel[j]);disc(ctx,p,8.5,colors[j]);ctx.font='bold 14px system-ui';ctx.fillStyle='#f7fbff';ctx.fillText('M'+(j+1),p[0]+12,p[1]-11);}
axes(ctx,viewL,w,h);ctx.fillStyle='#c7d8e4';ctx.font='12px system-ui';ctx.fillText('Centered global axes · mm',18,h-22);
let d=[];for(let j=0;j<4;j++){let p=rel[j],q=rel[(j+1)%4],dist=Math.hypot(...p.map((v,k)=>v-q[k]));d.push(`M${j+1}–M${(j+1)%4+1}: ${dist.toFixed(1)} mm`);}
document.getElementById('distance').textContent=d.join(' · ');}
function draw(){paintGlobal();paintLocal();document.getElementById('timeLabel').textContent=at.toFixed(1)+' s';uiTime.value=at;}
function pointerControls(canvas,view){let pressed=false,x0=0,y0=0;canvas.addEventListener('pointerdown',e=>{pressed=true;x0=e.clientX;y0=e.clientY;canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!pressed)return;view.yaw+=(e.clientX-x0)*.008;view.pitch=Math.max(-1.52,Math.min(1.52,view.pitch+(e.clientY-y0)*.006));x0=e.clientX;y0=e.clientY;draw()});
canvas.addEventListener('pointerup',()=>pressed=false);canvas.addEventListener('pointercancel',()=>pressed=false);
canvas.addEventListener('wheel',e=>{e.preventDefault();view.zoom=Math.max(.30,Math.min(5,view.zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});}
pointerControls(canvasG,viewG);pointerControls(canvasL,viewL);
document.getElementById('resetG').onclick=()=>{Object.assign(viewG,{yaw:-.68,pitch:.57,zoom:1.02});draw()};
document.getElementById('resetL').onclick=()=>{Object.assign(viewL,{yaw:-.73,pitch:.65,zoom:1});draw()};
document.getElementById('trails').onchange=e=>{showTrails=e.target.checked;draw()};
uiTime.oninput=()=>{at=parseFloat(uiTime.value);draw()};
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Ⅱ Pause':'▶ Play';lastFrame=0;requestAnimationFrame(tick)};
function tick(tm){if(!playing)return;if(lastFrame)at+=Math.min(.1,(tm-lastFrame)/1000)*Number(document.getElementById('speed').value);lastFrame=tm;
if(at>=608.05){at=608.05;playing=false;document.getElementById('play').textContent='▶ Play'}draw();if(playing)requestAnimationFrame(tick)}
window.addEventListener('resize',draw);draw();
</script></body></html>'''


def make_html(t, marks):
    # Preview uses 0.5-s frames; full 100-Hz data remains in the input folder.
    inds = np.r_[np.arange(0, len(t), 50), len(t) - 1]
    inds = np.unique(inds)
    sample = [[round(float(t[i]), 4),
               [[round(float(v / 1000), 4) for v in p] for p in marks[i]]]
              for i in inds]
    content = HTML.replace("__FRAME_DATA__", json.dumps(sample, separators=(",", ":")))
    (OUTPUT / "Task1_3D_Interactive.html").write_text(content, encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    t, marks, ref = load_task1()
    make_png(t, marks, ref)
    make_html(t, marks)
    print(f"Frames={len(t)}, duration={t[-1]} s, preview_frames={len(np.arange(0,len(t),50))+1}")
    print(f"PNG: {OUTPUT / 'Task1_3D_Overview.png'}")
    print(f"HTML: {OUTPUT / 'Task1_3D_Interactive.html'}")


if __name__ == "__main__":
    main()
