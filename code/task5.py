"""Task 5: gravity removal, band-pass filtering and trajectory comparison."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from task_common import G, OUTPUT, csv_xyz


def task5(t, f_ol, imu_r, omc_pos):
    out=OUTPUT;out.mkdir(parents=True, exist_ok=True)
    # At rest specific force in OG equals [0,0,+g].
    global_specific=np.einsum('nij,nj->ni',imu_r,f_ol)
    linear=global_specific-np.array([0.,0.,G])
    fs=1/np.median(np.diff(t))
    def integrate(low,high):
        sos=butter(4,[low,high],btype='bandpass',fs=fs,output='sos')
        filtered=sosfiltfilt(sos,linear,axis=0)
        vel=cumulative_trapezoid(filtered,t,axis=0,initial=0.)
        pos=omc_pos[0]+1000*cumulative_trapezoid(vel,t,axis=0,initial=0.)
        return pos,vel,filtered
    def metric(est,truth):
        err=est-truth
        return {'vector_rmse_mm':float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),
                'axis_rmse_mm':np.sqrt(np.mean(err**2,axis=0)).tolist(),
                'final_error_mm':float(np.linalg.norm(err[-1]))}
    fixed={}
    for low,high in [(0.3,5.),(0.5,7.),(1.1,10.)]:
        label=f'{low:g}_{high:g}Hz'
        est,vel,filt=integrate(low,high)
        fixed[label]=metric(est,omc_pos)
        csv_xyz(out/f'trajectory_{label}.csv',t,imu_position_og_mm=est,omc_centroid_og_mm=omc_pos)
        fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True)
        for j,ax in enumerate(axs):
            ax.plot(t,omc_pos[:,j],label='OMC marker centroid',lw=1,color='black')
            ax.plot(t,est[:,j],label='IMU integrated',lw=.8,color='#d35400')
            ax.set_ylabel('XYZ'[j]+' (mm)');ax.grid(alpha=.2)
        axs[0].legend();axs[0].set_title('Task 5(a): '+label)
        axs[-1].set_xlabel('OMC time (s)');fig.tight_layout()
        fig.savefig(out/f'trajectory_{label}.png',dpi=125);plt.close(fig)
    # OMC-only periodicity selection (no tuning against IMU-OMC RMSE).
    candidates=[]
    for start in np.arange(t[0]+10,t[-1]-31,10):
        ix=(t>=start)&(t<start+30)
        z=omc_pos[ix,2]
        freqs,pwr=welch(z-z.mean(),fs=fs,nperseg=min(len(z),2048))
        band=(freqs>=.4)&(freqs<=2.5)
        peak=freqs[np.flatnonzero(band)[np.argmax(pwr[band])]]
        concentration=pwr[(freqs>=peak-.12)&(freqs<=peak+.12)].sum()/max(pwr[freqs>=.3].sum(),1e-15)
        candidates.append((float(start),float(start+30),float(peak),float(concentration),float(z.std())))
    eligible=[c for c in candidates if c[4]>10.]
    selected=max(eligible or candidates,key=lambda x:x[3])
    lo,hi=selected[:2]
    ix=(t>=lo)&(t<hi);st=t[ix];sp=omc_pos[ix]
    f,pwr=welch(sp[:,2]-sp[:,2].mean(),fs=fs,nperseg=min(2048,len(st)))
    band=(f>=.4)&(f<=2.5)
    peak=float(f[np.flatnonzero(band)[np.argmax(pwr[band])]])
    smoothed=gaussian_filter1d(sp[:,2],3)
    peaks,_=find_peaks(smoothed,distance=int(.7*fs/peak),prominence=max(5,.3*np.std(smoothed)))
    diffs=np.diff(st[peaks])
    accepted=diffs[(diffs>.7/peak)&(diffs<1.4/peak)]
    T=float(np.median(accepted)) if len(accepted)>=3 else 1/peak
    f0=1/T
    low,high=round(.3*f0,3),round(min(max(3*f0,3),10),3)
    period_est,_,_=integrate(low,high)
    fixed_est,_,_=integrate(.3,5.)
    comparisons={'period_informed':metric(period_est[ix],sp),'fixed_0.3_5':metric(fixed_est[ix],sp)}
    csv_xyz(out/'period_comparison.csv',st,imu_period_mm=period_est[ix],imu_fixed_mm=fixed_est[ix],omc_mm=sp)
    fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True)
    for j,ax in enumerate(axs):
        ax.plot(st,sp[:,j],label='OMC',color='black')
        ax.plot(st,period_est[ix,j],label=f'Informed {low}-{high} Hz',lw=.8)
        ax.plot(st,fixed_est[ix,j],label='Fixed 0.3-5 Hz',lw=.8)
        ax.set_ylabel('XYZ'[j]+' (mm)');ax.grid(alpha=.2)
    axs[0].legend();axs[0].set_title('Task 5(b): same periodic segment')
    axs[-1].set_xlabel('OMC time (s)');fig.tight_layout();fig.savefig(out/'period_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,5))
    axs[0].plot(st,smoothed);axs[0].plot(st[peaks],smoothed[peaks],'x')
    axs[0].set(ylabel='OMC centroid Z (mm)',xlabel='Time (s)')
    axs[1].plot(f,pwr);axs[1].axvline(f0,color='red',label=f'f0 = {f0:.3f} Hz')
    axs[1].set(xlabel='Frequency (Hz)',ylabel='PSD (mm²/Hz)',xlim=(0,10));axs[1].legend()
    fig.tight_layout();fig.savefig(out/'period_selection.png',dpi=150);plt.close(fig)
    pd.DataFrame(candidates,columns=['start','end','peak_hz','spectral_concentration','std_z_mm']).to_csv(out/'period_candidates.csv',index=False)
    constant_baseline_mm = float(np.sqrt(np.mean(np.sum((omc_pos - omc_pos[0])**2,axis=1))))
    return {'fixed_band_results':fixed,'constant_position_baseline_RMSE_mm':constant_baseline_mm,
            'period_segment_s':[lo,hi],
            'period_T_s':T,'period_f0_hz':float(f0),'period_cutoffs_hz':[low,high],
            'period_valid_peak_intervals':int(len(accepted)),
            'period_comparison_RMSE':comparisons,
            'filter':'4th-order Butterworth bandpass, SOS filtfilt on global gravity-removed acceleration',
            'integration':'trapezoidal with initial velocity=0 and position=first OMC marker centroid',
            'point_assumption':'IMU at marker centroid (unknown lever arm); NOT physically verified',
            'rhythm':'OMC-derived same-foot centroid height cycle; intended spoken cadence unavailable'}
