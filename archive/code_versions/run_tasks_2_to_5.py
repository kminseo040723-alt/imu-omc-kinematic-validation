"""Reproduce Tasks 2-5. Run from any directory; defaults resolve beside this package.
Python >=3.9; numpy pandas scipy matplotlib. Task 1 output must exist.
Calibration: affine clocks, signed channel mapping and constant gyro bias from OMC.
Trajectory comparison assumes IMU origin coincides with marker centroid unless
--lever-arm-mm supplies centroid->IMU vector in empirical cluster coordinates.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.optimize import minimize
from scipy.signal import butter,sosfiltfilt,welch,find_peaks
from scipy.integrate import cumulative_trapezoid

ROOT=Path(__file__).resolve().parents[1]

def save(path,t,arrays):
 d={'time_s':t}
 for prefix,a in arrays.items():
  a=np.asarray(a)
  if a.ndim==1:d[prefix]=a
  else:
   for j in range(a.shape[1]):d[f'{prefix}_{j}']=a[:,j]
 pd.DataFrame(d).to_csv(path,index=False)

def plot_xyz(path,t,a,b,title,unit='mm'):
 fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
 for j,ax in enumerate(axs):
  ax.plot(t,b[:,j],label='OMC reference',lw=1);ax.plot(t,a[:,j],label='IMU estimate',lw=.8)
  ax.set_ylabel('XYZ'[j]+' ('+unit+')');ax.grid(alpha=.25)
 axs[0].legend();axs[0].set_title(title);axs[-1].set_xlabel('OMC export-relative time (s)');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=ROOT)
 ap.add_argument('--lever-arm-mm',type=float,nargs=3,default=[0,0,0])
 ap.add_argument('--period-start',type=float,default=None)
 ap.add_argument('--period-end',type=float,default=None)
 args=ap.parse_args();root=args.root
 out=root/'results';[ (out/f'task{k}').mkdir(parents=True,exist_ok=True) for k in range(2,6)]
 df=pd.read_csv(out/'task1/omc_refined.csv');t=df.omc_time.to_numpy();fs=1/np.median(np.diff(t))
 p=df[[f'marker_{m}_{a}' for m in range(1,5) for a in 'XYZ']].to_numpy().reshape(-1,4,3)
 if not np.isfinite(p).all():raise ValueError('Unresolved Task 1 coordinates. Select a complete segment before running this pipeline.')
 # OMC differentiation preprocessing. Original Task 1 coordinates remain saved.
 smooth=gaussian_filter1d(median_filter(p,size=(5,1,1)),3,axis=0)
 template=np.loadtxt(out/'task1/empirical_template_mm.csv',delimiter=',');template-=template.mean(0)
 centered=smooth-smooth.mean(1)[:,None,:]
 u,_,vt=np.linalg.svd(np.einsum('mi,nmj->nij',template,centered));D=np.tile(np.eye(3),(len(t),1,1));D[:,-1,-1]=np.linalg.det(u@vt)
 R=np.transpose(u@D@vt,(0,2,1)) # column-vector: cluster local -> OMC global
 fit=np.einsum('nij,mj->nmi',R,template)+smooth.mean(1)[:,None,:]
 residual=np.sqrt(np.mean(np.sum((fit-smooth)**2,axis=2),axis=1))
 tw=(t[1:]+t[:-1])/2
 wo=Rotation.from_matrix(np.transpose(R[:-1],(0,2,1))@R[1:]).as_rotvec()/np.diff(t)[:,None]
 im=pd.read_csv(root/'data/imu.csv');raw_time=im.imu_time.to_numpy();ti=raw_time-raw_time[0]
 gyro=im[['gx','gy','gz']].to_numpy()*np.pi/180;acc=im[['ax','ay','az']].to_numpy()*9.8
 def interp(q,data):return np.stack([np.interp(q,ti,data[:,j]) for j in range(data.shape[1])],axis=1)
 on=gaussian_filter1d(np.linalg.norm(wo,axis=1),5);gn=np.linalg.norm(gyro,axis=1)
 # Local offsets detect clock drift; search +/-30 s, require 90% overlap of each window.
 windows=[]
 for lo in np.arange(0,600,120):
  q=tw[(tw>=lo)&(tw<lo+120)][::10];y=np.interp(q,tw,on)
  scores=[];lags=np.arange(-30,30.01,.05)
  for lag in lags:
   v=(q-lag>=0)&(q-lag<=ti[-1]);scores.append(np.corrcoef(y[v],np.interp(q[v]-lag,ti,gn))[0,1] if v.sum()>.9*len(q) else -2)
  k=np.argmax(scores);windows.append([float(q.mean()),float(lags[k]),float(scores[k])])
 windows=np.array(windows);slope,intercept=np.polyfit(windows[:,0],windows[:,1],1)
 alpha0=1/(1-slope);beta0=alpha0*intercept
 q=tw[::5];y=on[::5]
 def objective(v):
  a,b=v
  if not .98<a<1.02 or abs(b)>40:return 2
  qt=(q-b)/a;valid=(qt>=0)&(qt<=ti[-1])
  if valid.sum()<.9*len(q):return 2
  return -np.corrcoef(y[valid],np.interp(qt[valid],ti,gn))[0,1]
 opt=minimize(objective,[alpha0,beta0],method='Nelder-Mead',options={'xatol':1e-10,'maxiter':1500})
 alpha,beta=opt.x
 pd.DataFrame(windows,columns=['window_center_s','offset_s','correlation']).to_csv(out/'task3/window_offsets.csv',index=False)
 # Determine signed gyro channel map from first 120 s; reserve later times for validation.
 g=interp((tw-beta)/alpha,gyro);cal=(tw>=5)&(tw<120)&(residual[:-1]<5)&(np.linalg.norm(wo,axis=1)<4.5)
 gc=g[cal].mean(0);wc=wo[cal].mean(0)
 u,_,vt=np.linalg.svd((g[cal]-gc).T@(wo[cal]-wc));M=(u@vt).T
 bias=gc-wc@M
 # C converts EUN channel ordering to ENU; Q is a proper mounting rotation.
 C=np.array([[1,0,0],[0,0,1],[0,1,0]],float);Q=M@C.T
 if np.linalg.det(Q)<0:raise ValueError('Observed channel convention differs from expected EUN reflection; inspect calibration.')
 aligned=(g-bias)@M.T;val=tw>=120
 err=np.sqrt(np.mean((aligned[val]-wo[val])**2,axis=0))*180/np.pi
 np.savetxt(out/'task2/axis_conversion_C.csv',C,delimiter=',');np.savetxt(out/'task2/mounting_rotation_Q.csv',Q,delimiter=',')
 np.savetxt(out/'task2/raw_gyro_to_cluster_M.csv',M,delimiter=',')
 save(out/'task2/imu_aligned.csv',ti,{'gyro_cluster_rad_s':(gyro-bias)@M.T,'specific_force_cluster_m_s2':acc@M.T})
 save(out/'task3/omc_ang_vel.csv',tw,{'omega_cluster_rad_s':wo})
 save(out/'task3/imu_ang_vel.csv',alpha*ti+beta,{'omega_cluster_rad_s':(gyro-bias)@M.T})
 fig,axs=plt.subplots(2,1,figsize=(11,6));sel=(tw>280)&(tw<290)
 axs[0].plot(tw[sel],np.linalg.norm(wo[sel],axis=1),label='OMC');axs[0].plot(tw[sel],np.interp(tw[sel],ti,gn),label='IMU, no clock alignment');axs[0].legend()
 axs[1].plot(tw[sel],np.linalg.norm(wo[sel],axis=1),label='OMC');axs[1].plot(tw[sel],np.interp((tw[sel]-beta)/alpha,ti,gn),label='IMU, affine clock alignment');axs[1].legend()
 for ax in axs:ax.set_ylabel('Angular speed (rad/s)');ax.grid(alpha=.2)
 axs[-1].set_xlabel('OMC time (s)');fig.tight_layout();fig.savefig(out/'task3/time_alignment.png',dpi=150);plt.close(fig)
 # Common 100 Hz interval, no extrapolation.
 keep=(t>=max(t[0],beta))&(t<=min(t[-1],alpha*ti[-1]+beta));tt=t[keep];rr=R[keep]
 aa=interp((tt-beta)/alpha,acc)@M.T;gg=(interp((tt-beta)/alpha,gyro)-bias)@M.T
 positions=smooth.mean(1)[keep]+np.einsum('nij,j->ni',rr,np.asarray(args.lever_arm_mm))
 save(out/'task3/imu_synched.csv',tt,{'specific_force_cluster_m_s2':aa,'gyro_cluster_rad_s':gg})
 save(out/'task3/omc_synched.csv',tt,{'point_mm':positions,'rigid_fit_residual_mm':residual[keep]})
 # Quaternion xyzw, active cluster->global, right multiplication for body gyro.
 rotations=np.empty((len(tt),3,3));orientation=Rotation.from_matrix(rr[0]);rotations[0]=rr[0]
 for k in range(1,len(tt)):
  orientation=orientation*Rotation.from_rotvec(.5*(gg[k-1]+gg[k])*(tt[k]-tt[k-1]));rotations[k]=orientation.as_matrix()
 qi=Rotation.from_matrix(rotations).as_quat();qo=Rotation.from_matrix(rr).as_quat()
 for quat in [qi,qo]:
  for k in range(1,len(quat)):
   if np.dot(quat[k],quat[k-1])<0:quat[k]*=-1
 angle=Rotation.from_matrix(np.transpose(rr,(0,2,1))@rotations).magnitude()*180/np.pi
 save(out/'task4/imu_orientation.csv',tt,{'quaternion_xyzw':qi});save(out/'task4/omc_orientation.csv',tt,{'quaternion_xyzw':qo});save(out/'task4/orientation_error.csv',tt,{'relative_rotation_deg':angle})
 fig,ax=plt.subplots(figsize=(11,4));ax.plot(tt,angle);ax.axvspan(5,120,color='orange',alpha=.15,label='OMC-aided gyro calibration');ax.legend();ax.set(xlabel='Time (s)',ylabel='Relative rotation error (deg)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'task4/orientation_error.png',dpi=150);plt.close(fig)
 # Positive accelerometer reading at rest is +g after rotation: subtract [0,0,9.8].
 linear=np.einsum('nij,nj->ni',rotations,aa)-[0,0,9.8]
 gravity_check=np.mean(np.einsum('nij,nj->ni',rr,aa),axis=0)
 # Bandpass both removes low frequency components and distorts true translation.
 # No endpoint correction, ZUPT, trajectory scaling, or repeated OMC attitude correction.
 settings=[(.3,5),(.5,7),(1.1,10)];metrics=[]
 def estimate(a,time,low,high,origin):
  filtered=sosfiltfilt(butter(4,[low,high],btype='bandpass',fs=fs,output='sos'),a,axis=0)
  velocity=cumulative_trapezoid(filtered,time,axis=0,initial=0)
  trajectory=origin+cumulative_trapezoid(velocity,time,axis=0,initial=0)*1000
  return trajectory,velocity,filtered
 def metric(name,est,ref):
  e=est-ref;return {'setting':name,'position_rmse_mm':float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),'axis_rmse_mm':np.sqrt(np.mean(e*e,axis=0)).tolist(),'final_error_mm':float(np.linalg.norm(e[-1]))}
 for low,high in settings:
  tag=f'{low:g}_{high:g}Hz';est,v,f=estimate(linear,tt,low,high,positions[0]);metrics.append(metric(tag,est,positions))
  save(out/f'task5/trajectory_{tag}.csv',tt,{'imu_mm':est,'omc_point_mm':positions,'velocity_m_s':v,'filtered_acc_m_s2':f})
  plot_xyz(out/f'task5/trajectory_{tag}.png',tt,est,positions,'Task 5(a): BPF '+tag)
 # Find periodic segment from OMC centroid Z only; choose by spectral concentration,
 # independent of IMU position error. Same-foot height cycle, not step count.
 candidates=[]
 for start in np.arange(tt[0]+10,tt[-1]-31,10):
  ix=(tt>=start)&(tt<start+30);zz=positions[ix,2];freq,power=welch(zz-zz.mean(),fs=fs,nperseg=min(len(zz),2048));band=(freq>=.4)&(freq<=2.5)
  k=np.flatnonzero(band)[np.argmax(power[band])];f0=freq[k];concentration=power[(freq>=f0-.12)&(freq<=f0+.12)].sum()/max(power[freq>=.3].sum(),1e-15)
  candidates.append([float(start),float(start+30),float(f0),float(concentration),float(np.std(zz))])
 eligible=[c for c in candidates if c[4]>10];chosen=max(eligible or candidates,key=lambda c:c[3])
 lo=args.period_start if args.period_start is not None else chosen[0];hi=args.period_end if args.period_end is not None else chosen[1]
 seg=(tt>=lo)&(tt<hi)
 if seg.sum()<200:raise ValueError('Periodic segment must contain >=200 samples.')
 st=tt[seg];sp=positions[seg];sl=linear[seg]
 freq,power=welch(sp[:,2]-sp[:,2].mean(),fs=fs,nperseg=min(len(st),2048));band=(freq>=.4)&(freq<=2.5);f0=float(freq[np.flatnonzero(band)[np.argmax(power[band])]])
 # Refine full-cycle period using successive peaks of the same foot's height.
 zz=gaussian_filter1d(sp[:,2],3);peaks,_=find_peaks(zz,distance=int(.7*fs/f0),prominence=max(5,.3*np.std(zz)))
 intervals=np.diff(st[peaks]);accepted=intervals[(intervals>.7/f0)&(intervals<1.4/f0)]
 T=float(np.median(accepted)) if len(accepted)>=3 else 1/f0;f0=1/T
 low=round(.3*f0,3);high=round(min(max(3*f0,3),10),3)
 compare=[]
 fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
 for j,ax in enumerate(axs):ax.plot(st,sp[:,j],label='OMC point',color='black')
 for label,l,h in [('fixed_0.3_5',.3,5),('period_informed',low,high)]:
  full_est,full_v,full_f=estimate(linear,tt,l,h,positions[0]);est=full_est[seg];v=full_v[seg];compare.append(metric(label,est,sp));save(out/f'task5/period_{label}.csv',st,{'imu_mm':est,'omc_point_mm':sp,'velocity_m_s':v})
  for j,ax in enumerate(axs):ax.plot(st,est[:,j],label=label);ax.set_ylabel('XYZ'[j]+' (mm)');ax.grid(alpha=.2)
 axs[0].legend();axs[0].set_title('Task 5(b): same segment, full-record filtering/integration unchanged');axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'task5/period_comparison.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(2,1,figsize=(10,6));axs[0].plot(st,zz);axs[0].plot(st[peaks],zz[peaks],'rx');axs[0].set(xlabel='Time (s)',ylabel='Foot centroid Z (mm)');axs[1].plot(freq,power);axs[1].axvline(f0,color='red',label=f'f0={f0:.3f} Hz');axs[1].set(xlim=(0,12),xlabel='Frequency (Hz)',ylabel='PSD (mm²/Hz)');axs[1].legend();fig.tight_layout();fig.savefig(out/'task5/period_selection.png',dpi=150);plt.close(fig)
 pd.DataFrame(candidates,columns=['start_s','end_s','spectral_peak_hz','concentration','z_std_mm']).to_csv(out/'task5/period_candidates.csv',index=False)
 summary={'clock':{'first_raw_imu_timestamp_s':float(raw_time[0]),'alpha':float(alpha),'beta_s':float(beta),'formula':'t_OMC = alpha * (imu_time - first_imu_time) + beta','norm_correlation':float(-opt.fun),'optimizer_success':bool(opt.success)},'calibration':{'interval_s':[5,120],'gyro_bias_raw_deg_s':(bias*180/np.pi).tolist(),'gyro_validation_axis_rmse_deg_s':err.tolist(),'M_det':float(np.linalg.det(M)),'Q_det':float(np.linalg.det(Q)),'OMC_aided_acc_global_mean_m_s2':gravity_check.tolist(),'gyro_convention':'Raw gyro signed channels empirically fit with the EUN permutation, then proper mounting Q. This is channel-convention calibration, not an assumed polar/axial vector rule.'},'orientation':{'quaternion_order':'xyzw','initialization':'first synchronized OMC orientation','mean_error_deg':float(angle.mean()),'last_error_deg':float(angle[-1]),'max_error_deg':float(angle.max())},'trajectory':{'point':'marker centroid + R_OMC @ lever_arm','lever_arm_mm':args.lever_arm_mm,'nonzero_lever_arm_supplied':bool(np.linalg.norm(args.lever_arm_mm)>0),'initial_velocity':'zero assumption','filter':'4th-order Butterworth prototype, forward-backward SOS, on global gravity-removed acceleration','fixed_results':metrics},'period':{'segment_s':[float(st[0]),float(st[-1])],'selection':'automatic OMC same-foot centroid height periodicity, not verified against instructed rhythm','full_cycle_T_s':T,'f0_hz':f0,'accepted_intervals':len(accepted),'cutoffs_hz':[low,high],'comparison':compare},'limitations':['Unknown physical IMU-to-marker-centroid offset: default zero is an unverified approximation. Supply measured lever arm for same-point validation.','Instruction rhythm and intended periodic segment were not provided; automatic candidate requires confirmation.','OMC coordinate preprocessing: 5-sample median then Gaussian sigma=3 samples; affects angular velocity.','Task 1 geometry suspects remain in source; smoothing is not proof of correction.','OMC-aided calibration 5-120 s; gyro validation metrics use times >=120 s, but clock fit uses full record.','Pure gyro integration has no ongoing OMC correction; resulting gravity leakage and double integration can be severe.','Single-record clock scaling estimated empirically; not established hardware clock diagnosis.']}
 (out/'summary_tasks2_5.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
