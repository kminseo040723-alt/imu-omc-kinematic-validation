"""MOBI311 sequential Tasks 2--5, reproducible analysis; Task 1 already completed.

Run: python mobi311_tasks2to5_gravity_only.py
Requires numpy, scipy, pandas, matplotlib.

Task 1 reconstruction is deliberately NOT run or overwritten. The available
OMC export has complete marker positions; its empirical reference template is
formed read-only at startup. Task 2 uses a single static gravity direction
to choose the minimum-angle tilt-only alignment. Heading is not identifiable;
neither photographs nor clock-matched dynamic signals enter this calibration.
Task 3 subsequently estimates clocks from angular speed norms.

All generated data, plots and a summary go into
mobi311_results_task2to5_gravity_only/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'mobi311_results_task2to5_gravity_only'
G = 9.8
FS_OMC = 100.
C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)


def csv_xyz(filename, t, **groups):
    cols = {'time_s': t}
    for name, x in groups.items():
        x = np.asarray(x)
        if x.ndim == 1:
            cols[name] = x
        else:
            for j in range(x.shape[1]):
                cols[f'{name}_{"xyz"[j] if x.shape[1] == 3 else j}'] = x[:, j]
    pd.DataFrame(cols).to_csv(filename, index=False)


def extract_raw():
    # Vicon ASCII export: first 5 rows are metadata, frame index in col 0.
    d = pd.read_csv(ROOT / 'upload/omc_vicon.csv', header=None, skiprows=5)
    frames = d.iloc[:, 0].to_numpy(float)
    t = (frames - frames[0]) / FS_OMC
    p = d.iloc[:, 2:14].to_numpy(float).reshape(-1, 4, 3)
    imu = pd.read_csv(ROOT / 'upload/imu(3).csv')
    ti = imu.imu_time.to_numpy(float)
    ti = ti - ti[0]
    gyro = imu[['gx', 'gy', 'gz']].to_numpy(float) * np.pi / 180.
    accel = imu[['ax', 'ay', 'az']].to_numpy(float) * G
    return t, p, ti, gyro, accel


def local_frame(p):
    """Empirical OL: +X marker 1 -> 2, +Y mostly 2 -> 3, +Z RH normal."""
    initial = p[:200].mean(axis=0)
    ex = initial[1] - initial[0]; ex /= np.linalg.norm(ex)
    ey = initial[2] - initial[1]
    ey = ey - ex * np.dot(ex, ey); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    b0 = np.column_stack((ex, ey, ez))  # column axes of OL in OG
    template = (initial - initial.mean(axis=0)) @ b0
    return b0, template


def rigid_fit(p, template):
    """Right-handed local->global rigid fit via proper Kabsch rotations."""
    ctr = p.mean(axis=1)
    cov = np.einsum('mi,nmj->nij', template, p - ctr[:, None, :])
    u, _, vt = np.linalg.svd(cov)
    d = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
    d[:, 2, 2] = np.linalg.det(u @ vt)
    rotation = np.transpose(u @ d @ vt, (0, 2, 1))
    fit = np.einsum('nij,mj->nmi', rotation, template) + ctr[:, None, :]
    residual = np.sqrt(np.mean(np.sum((fit-p)**2, axis=2), axis=1))
    return rotation, ctr, residual


def task1(t, p):
    out = OUTPUT / 'task1'; out.mkdir(parents=True, exist_ok=True)
    b0, template = local_frame(p)
    actual_missing = ~np.isfinite(p).all(axis=2)
    # The supplied Vicon export has no missing points. Preserve all of them.
    if np.any(actual_missing):
        raise ValueError('Unexpected missing markers: adapt the real-gap branch before use.')
    refined = p.copy()
    cols = {f'marker_{m}_{a}': refined[:, m - 1, j]
            for m in range(1, 5) for j, a in enumerate('XYZ')}
    pd.DataFrame({'omc_time': t, **cols}).to_csv(out / 'omc_refined.csv', index=False)
    np.savetxt(out / 'empirical_template_mm.csv', template, delimiter=',', fmt='%.8f')
    np.savetxt(out / 'initial_OL_to_OG.csv', b0, delimiter=',', fmt='%.8f')

    # Clearly marked *synthetic* dropout, not fabricated source missing values.
    masked = p.copy()
    partial = (t >= 180) & (t < 180.8)
    full = (t >= 185) & (t < 185.35)
    masked[partial, 0, :] = np.nan
    masked[full, :, :] = np.nan
    reconstruct = masked.copy()
    # >=3 visible markers: rigid fit; one missing marker can be reconstructed.
    use = partial
    sub = masked[use][:, 1:, :]
    sub_template = template[1:]
    sub_center = sub.mean(axis=1)
    u, _, vt = np.linalg.svd(np.einsum('mi,nmj->nij', sub_template - sub_template.mean(0), sub - sub_center[:, None, :]))
    dd = np.broadcast_to(np.eye(3), (len(sub), 3, 3)).copy()
    dd[:, 2, 2] = np.linalg.det(u @ vt)
    rr = np.transpose(u @ dd @ vt, (0, 2, 1))
    tx = sub_center - np.einsum('nij,j->ni', rr, sub_template.mean(0))
    reconstruct[use, 0, :] = np.einsum('nij,j->ni', rr, template[0]) + tx
    # No markers: interpolate R(t) with SLERP and translation linearly.
    lo = np.flatnonzero(t < 185)[-1]
    hi = np.flatnonzero(t >= 185.35)[0]
    raw_fit = gaussian_filter1d(p, 1, axis=0)
    all_rr, all_ctr, _ = rigid_fit(raw_fit, template)
    times = t[full]
    rotations = Slerp([t[lo], t[hi]], Rotation.from_matrix(all_rr[[lo, hi]]))(times).as_matrix()
    ctr = (1 - (times-t[lo])[:, None]/(t[hi]-t[lo]))*all_ctr[lo] + ((times-t[lo])[:, None]/(t[hi]-t[lo]))*all_ctr[hi]
    reconstruct[full] = np.einsum('nij,mj->nmi', rotations, template) + ctr[:, None, :]
    partial_error = float(np.sqrt(np.mean((reconstruct[partial, 0] - p[partial, 0])**2)))
    full_error = float(np.sqrt(np.mean(np.sum((reconstruct[full] - p[full])**2, axis=2))))
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=False)
    for ax, region, marker, name in [(axs[0], (t >= 179.5)&(t<=181.3), 0, 'One-marker simulated occlusion'),
                                      (axs[1], (t>=184.5)&(t<=186), 0, 'Four-marker simulated occlusion')]:
        ax.plot(t[region], p[region, marker, 2], '--', color='gray', lw=1, label='OMC ground truth (held out)')
        observed = region & np.isfinite(masked[:, marker, 2])
        missing = region & ~np.isfinite(masked[:, marker, 2])
        ax.scatter(t[observed], masked[observed, marker, 2], color='#2077a9', s=6, label='Observed')
        ax.plot(t[missing], reconstruct[missing, marker, 2], color='#d35400', lw=2, label='Reconstructed')
        ax.set(ylabel='Marker 1 Z (mm)', title=name)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8, ncol=3); axs[-1].set_xlabel('OMC time (s)')
    fig.tight_layout(); fig.savefig(out / 'occlusion_demonstration.png', dpi=150);plt.close(fig)
    return b0, template, {'original_missing_marker_samples': int(actual_missing.sum()),
                          'synthetic_one_marker_RMSE_per_axis_mm': partial_error,
                          'synthetic_full_gap_marker_vector_RMSE_mm': full_error,
                          'synthetic_gaps_s': [[180,180.8], [185,185.35]],
                          'note': 'All original markers observed; masking only for a validation demonstration.'}


def task2(t, ti, gyro, accel, b0):
    out = OUTPUT / 'task2';out.mkdir(exist_ok=True)
    # Static IMU 1--5 seconds: specific-force-up vector and raw gyro bias.
    static = (ti >= 1.) & (ti < 5.)
    mean_acc_enu = C @ accel[static].mean(0)
    sensed_up_enu = mean_acc_enu / np.linalg.norm(mean_acc_enu)
    target_up_ol = b0.T @ np.array([0., 0., 1.])
    # Exactly one direction pair: SciPy returns the shortest (minimum-angle)
    # rotation mapping the measured vector to the target vector. Yaw around
    # the target up-axis remains unobservable; this choice fixes a gauge,
    # not a measured physical mounting heading.
    mount, _ = Rotation.align_vectors(target_up_ol, sensed_up_enu)
    Q = mount.as_matrix()  # converted IMU ENU -> marker local OL (proper)
    M = Q @ C           # raw stored EUN channels -> OL (orthogonal, improper)
    bias = gyro[static].mean(0)
    w_ol = (gyro - bias) @ M.T
    f_ol = accel @ M.T
    np.savetxt(out/'eun_to_enu_C.csv', C, delimiter=',', fmt='%.8f')
    np.savetxt(out/'enu_to_ol_Q.csv', Q, delimiter=',', fmt='%.8f')
    np.savetxt(out/'raw_to_ol_M.csv', M, delimiter=',', fmt='%.8f')
    csv_xyz(out/'imu_aligned.csv', ti, gyro_ol_rad_s=w_ol, specific_force_ol_m_s2=f_ol)
    gfit = Q @ sensed_up_enu
    resid = np.degrees(np.arccos(np.clip(np.dot(gfit, target_up_ol), -1, 1)))
    return M, w_ol, f_ol, {'calibration': 'Static 1--5 s gravity-only shortest-rotation initial alignment; yaw unobservable; no time synchronization',
                           'static_imu_interval_s':[1.,5.], 'static_bias_raw_deg_s':(bias*180/np.pi).tolist(),
                           'C':C.tolist(), 'Q':Q.tolist(), 'M':M.tolist(),
                           'det_C':float(np.linalg.det(C)), 'det_Q':float(np.linalg.det(Q)),
                           'det_M':float(np.linalg.det(M)),
                           'gravity_angular_residual_deg_IN_SAMPLE':float(resid),
                           'minimum_angle_rotation_deg':float(np.degrees(mount.magnitude())),
                           'mean_static_specific_force_norm_m_s2':float(np.linalg.norm(mean_acc_enu)),
                           'imu_static_up_unit_ENU':sensed_up_enu.tolist(),
                           'omc_static_up_unit_OL':target_up_ol.tolist(),
                           'heading_or_yaw_about_up':'Underdetermined; minimum-angle gauge selection, NOT independently calibrated',
                           'photograph_use':'Only physical marker placement and mounting documentation; never used for axis alignment',
                           'axis_sign_assumption':'CSV gyro and accel channel signs are operationally transformed by same M; firmware not verified',
                           'accelerometer_specific_force':'At rest +9.8 m/s^2 toward global up; not linear acceleration'}


def imuint(tquery, tsource, values):
    return np.column_stack([np.interp(tquery, tsource, values[:, k]) for k in range(values.shape[1])])


def task3(t, p, template, ti, w_ol, f_ol):
    out=OUTPUT/'task3';out.mkdir(exist_ok=True)
    smooth=gaussian_filter1d(median_filter(p, size=(5,1,1)), 3, axis=0)
    r, pos, residual=rigid_fit(smooth,template)
    tw=(t[:-1]+t[1:]) / 2
    # Body angular velocity: Log( R_k^T R_k+1 ) / dt, resolved in OL.
    omc_w=Rotation.from_matrix(np.transpose(r[:-1],(0,2,1))@r[1:]).as_rotvec()/np.diff(t)[:,None]
    omc_speed=gaussian_filter1d(np.linalg.norm(omc_w,axis=1),5)
    imu_speed=np.linalg.norm(w_ol,axis=1)
    # Clocks: compare scalar angular speed (invariant to any fixed orthogonal Q).
    lag_list=[]
    for start in np.arange(0,600,120):
        q=tw[(tw>=start)&(tw<start+120)][::10]
        if not len(q):continue
        y=np.interp(q,tw,omc_speed)
        shifts=np.arange(-30,30.001,.05)
        corr=[]
        for lag in shifts:
            valid=(q-lag>=0)&(q-lag<=ti[-1])
            corr.append(np.corrcoef(y[valid],np.interp(q[valid]-lag,ti,imu_speed))[0,1]
                        if valid.sum()>.9*len(q) else -2)
        k=np.argmax(corr);lag_list.append((float(q.mean()),float(shifts[k]),float(corr[k])))
    lags=np.asarray(lag_list)
    slope,intercept=np.polyfit(lags[:,0],lags[:,1],1)
    a0,b0=1/(1-slope),intercept/(1-slope)
    q=tw[::5];y=omc_speed[::5]
    def objective(v):
        a,b=v
        if not .98<a<1.02 or abs(b)>40:return 2.
        mapped=(q-b)/a
        valid=(mapped>=0)&(mapped<=ti[-1])
        if valid.sum()<.9*len(q):return 2.
        return -np.corrcoef(y[valid],np.interp(mapped[valid],ti,imu_speed))[0,1]
    opt=minimize(objective,[a0,b0],method='Nelder-Mead',options={'xatol':1e-10,'maxiter':1500})
    alpha,beta=opt.x
    pd.DataFrame(lags,columns=['omc_window_center_s','lag_omc_minus_imu_s','correlation']).to_csv(out/'clock_lags.csv',index=False)
    csv_xyz(out/'omc_ang_vel.csv',tw,omega_ol_rad_s=omc_w)
    csv_xyz(out/'imu_ang_vel.csv',alpha*ti+beta,omega_ol_rad_s=w_ol)
    sel=(tw>=280)&(tw<291)
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
    axes[0].plot(tw[sel],omc_speed[sel],label='OMC')
    axes[0].plot(tw[sel],np.interp(tw[sel],ti,imu_speed),label='IMU no correction')
    axes[1].plot(tw[sel],omc_speed[sel],label='OMC')
    axes[1].plot(tw[sel],np.interp((tw[sel]-beta)/alpha,ti,imu_speed),label='IMU synchronized')
    for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set_ylabel('Angular speed (rad/s)')
    axes[1].set_xlabel('OMC time (s)');fig.tight_layout();fig.savefig(out/'synchronization.png',dpi=150);plt.close(fig)
    # 100 Hz common grid. Never extrapolate past overlap.
    mask=(t>=max(t[0],beta))&(t<=min(t[-1],alpha*ti[-1]+beta))
    sync_t=t[mask]; query=(sync_t-beta)/alpha
    sync_f=imuint(query,ti,f_ol); sync_w=imuint(query,ti,w_ol)
    csv_xyz(out/'imu_synched.csv',sync_t,gyro_ol_rad_s=sync_w,specific_force_ol_m_s2=sync_f)
    csv_xyz(out/'omc_synched.csv',sync_t,centroid_og_mm=pos[mask],rigid_fit_residual_mm=residual[mask])
    # Out-of-calibration independent spatial check, ONLY after time synchronization.
    query_w=(tw-beta)/alpha
    valid=(tw>=120)&(tw<=min(t[-1],alpha*ti[-1]+beta))&(query_w>=0)&(query_w<=ti[-1])
    aligned=imuint(query_w[valid],ti,w_ol)
    er=aligned-omc_w[valid]
    vector_rmse=float(np.sqrt(np.mean(np.sum(er**2,axis=1)))*180/np.pi)
    # A gyro pseudovector would change sign under a *literal* handedness
    # reflection. Compare both operational CSV sign hypotheses after Task 3.
    opposite_sign_rmse=float(np.sqrt(np.mean(np.sum((-aligned-omc_w[valid])**2,axis=1)))*180/np.pi)
    omc_vector_rms=float(np.sqrt(np.mean(np.sum(omc_w[valid]**2,axis=1)))*180/np.pi)
    fast=np.linalg.norm(omc_w[valid],axis=1)>.5
    dot=np.sum(aligned[fast]*omc_w[valid][fast],axis=1)
    angle=np.degrees(np.arccos(np.clip(dot/(np.linalg.norm(aligned[fast],axis=1)*np.linalg.norm(omc_w[valid][fast],axis=1)+1e-12),-1,1)))
    return sync_t,sync_w,sync_f,r[mask],pos[mask],{
        'clock_model':'t_OMC = alpha * (t_IMU - t_IMU_first) + beta',
        'alpha':float(alpha),'beta_s':float(beta),'angular_speed_correlation':float(-opt.fun),
        'overlap_s':[float(sync_t[0]),float(sync_t[-1])],
        'posthoc_independent_calibration_gyro_vector_rmse_deg_s':vector_rmse,
        'posthoc_opposite_all_gyro_sign_rmse_deg_s':opposite_sign_rmse,
        'posthoc_reference_omc_gyro_rms_deg_s':omc_vector_rms,
        'posthoc_gyro_direction_median_error_deg':float(np.median(angle)),
        'posthoc_eval_interval_omc_s':[120.,600.],
        'note':'OMC-gyro pairing is validation only; Task 2 Q, M, and static bias were never refitted.'}


def task4(t, w, omc_r):
    out=OUTPUT/'task4';out.mkdir(exist_ok=True)
    # Active quaternion represents OL->OG, scalar-last [qx,qy,qz,qw].
    # Right multiply body-frame increment. Initialize using first OMC pose once.
    n=len(t);estimated=np.empty((n,4))
    ori=Rotation.from_matrix(omc_r[0]);estimated[0]=ori.as_quat()
    for k in range(1,n):
        ori=ori*Rotation.from_rotvec((w[k-1]+w[k])*.5*(t[k]-t[k-1]))
        estimated[k]=ori.as_quat()
    measured=Rotation.from_matrix(omc_r).as_quat()
    for q in [estimated,measured]:
        for k in range(1,len(q)):
            if np.dot(q[k],q[k-1])<0:q[k]*=-1.
    rel=(Rotation.from_quat(measured).inv()*Rotation.from_quat(estimated)).magnitude()
    angle=np.degrees(rel)
    csv_xyz(out/'imu_orientation.csv',t,quat_xyzw=estimated)
    csv_xyz(out/'omc_orientation.csv',t,quat_xyzw=measured)
    csv_xyz(out/'orientation_error.csv',t,angle_deg=angle)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(t,angle,color='#ae4938',lw=.8)
    ax.set(xlabel='OMC time (s)',ylabel='Relative rotation error (degrees)',title='Task 4: IMU vs OMC orientation')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'orientation_error.png',dpi=150);plt.close(fig)
    return Rotation.from_quat(estimated).as_matrix(),{
        'quaternion_order':'[qx,qy,qz,qw], active local OL -> global OG',
        'initial_condition':'OMC first synchronized orientation at overlap start',
        'integration':'q[k+1]=q[k] * exp(0.5*(omega[k]+omega[k+1])*dt), body-frame right product',
        'mean_error_deg':float(angle.mean()),'median_error_deg':float(np.median(angle)),
        'final_error_deg':float(angle[-1]),'max_error_deg':float(angle.max()),
        'first_30_s_mean_error_deg':float(angle[t<t[0]+30].mean())}


def task5(t, f_ol, imu_r, omc_pos):
    out=OUTPUT/'task5';out.mkdir(exist_ok=True)
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
    return {'fixed_band_results':fixed,'period_segment_s':[lo,hi],
            'period_T_s':T,'period_f0_hz':float(f0),'period_cutoffs_hz':[low,high],
            'period_valid_peak_intervals':int(len(accepted)),
            'period_comparison_RMSE':comparisons,
            'filter':'4th-order Butterworth bandpass, SOS filtfilt on global gravity-removed acceleration',
            'integration':'trapezoidal with initial velocity=0 and position=first OMC marker centroid',
            'point_assumption':'IMU at marker centroid (unknown lever arm); NOT physically verified',
            'rhythm':'OMC-derived same-foot centroid height cycle; intended spoken cadence unavailable'}


def main():
    OUTPUT.mkdir(exist_ok=True)
    t,p,ti,gyro,accel=extract_raw()
    # Task 1 is the user's completed prior step. Load its saved reference
    # geometry as a fixed read-only input; do not rebuild or overwrite Task 1.
    if not np.isfinite(p).all():
        raise ValueError('Provide completed Task 1 refined OMC data; source export has missing markers.')
    task1_ref=ROOT/'mobi311_results'/'task1'
    b0=np.loadtxt(task1_ref/'initial_OL_to_OG.csv',delimiter=',')
    template=np.loadtxt(task1_ref/'empirical_template_mm.csv',delimiter=',')
    if b0.shape!=(3,3) or template.shape!=(4,3):
        raise ValueError('Saved Task 1 reference geometry has unexpected dimensions')
    M,w,f,s2=task2(t,ti,gyro,accel,b0)
    ts,ws,fs,rs,ps,s3=task3(t,p,template,ti,w,f)
    rimu,s4=task4(ts,ws,rs)
    s5=task5(ts,fs,rimu,ps)
    summary={'metadata':{'omc_frames':len(t),'omc_duration_s':float(t[-1]),
                         'imu_samples':len(ti),'imu_duration_s':float(ti[-1]),
                         'omc_frame':'right-handed global ENU, mm',
                         'imu_frame':'recorded left-handed local EUN, deg/s & g'},
             'task2':s2,'task3':s3,'task4':s4,'task5':s5,
             'limitations':['Photographs document mounting location and marker identification only; they are not calibration observations.',
                            'A single static gravity vector cannot determine rotation about vertical; the minimum-angle solution is an explicit gauge convention, not full mounting calibration.',
                            'Accelerometer and gyro same signed mapping is assumed and tested only after Task 3.',
                            'Task 1 is considered completed externally; this run does not alter or reconstruct marker samples.',
                            'Foot sensor location unknown; OMC marker centroid assumed co-located for Task 5.',
                            'No ZUPT, lever-arm correction, heading reference, or drift-reset used.',
                            'Rhythm inferred from OMC height, not confirmed using experimental instruction.',
                            'Task 4 OMC only used at initial synchronized pose, not during gyroscope integration.']}
    (OUTPUT/'results_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ['metadata','task2','task3','task4','task5']},indent=2))


if __name__=='__main__':
    main()
