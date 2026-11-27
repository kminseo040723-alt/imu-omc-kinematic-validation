"""MOBI311 Tasks 2--5 with synchronized dynamic SVD calibration.

Run: python code/run_tasks2to5.py (from the project root or any directory).
Requires numpy, scipy, pandas, matplotlib.

The already completed Task 1 OMC refinement and reference geometry are
supplied as READ-ONLY input copies. No Task 1 result is recomputed.
Chronology: time sync from scalar angular-speed norms (without a fitted Q),
then fixed Q and gyro bias fitted using paired 3-axis angular rates in
OMC 5--120 s; 120--600 s is held out; quaternion orientation and filtered
position estimation follow. All outputs are written to ../results/.
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
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results'
RAW = ROOT / 'raw_csv'
OUTPUT = ROOT / 'results'
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
    # Import the user's already refined Task 1 OMC file without changing it.
    d = pd.read_csv(INPUT / 'omc_refined_task1.csv')
    t = d.omc_time.to_numpy(float)
    p = np.stack([d[[f'marker_{m}_{j}' for j in 'XYZ']].to_numpy(float)
                  for m in range(1,5)],axis=1)
    imu = pd.read_csv(RAW / 'imu.csv')
    ti = imu.imu_time.to_numpy(float)
    ti = ti - ti[0]
    gyro = imu[['gx', 'gy', 'gz']].to_numpy(float) * np.pi / 180.
    accel = imu[['ax', 'ay', 'az']].to_numpy(float) * G
    return t, p, ti, gyro, accel


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


def task2(ti, gyro, accel, b0, alpha, beta, tw, omc_w, fit_residual):
    """Fixed local mapping and gyro bias from Task 3 synchronized 3-axis rates.

    The 5--120-s OMC samples determine Q and a constant bias. A disjoint
    120--600-s set validates the fit. The preferred +/- gyro channel parity
    is selected using the calibration data ONLY. Acceleration uses Q*C
    independently of that gyro pseudovector/sign hypothesis.
    """
    out = OUTPUT; out.mkdir(parents=True, exist_ok=True)
    static = (ti >= 1.) & (ti < 5.)
    bias_static = gyro[static].mean(axis=0)
    target_up = b0.T @ np.array([0., 0., 1.])
    f_static = accel[static].mean(axis=0)
    q_tilt, _ = Rotation.align_vectors(target_up, C @ f_static)
    w_baseline = (gyro - bias_static) @ (q_tilt.as_matrix() @ C).T
    query = (tw - beta) / alpha
    valid = (query >= ti[0]) & (query <= ti[-1])
    good_omc = (fit_residual[:-1] + fit_residual[1:]) / 2 < 5
    angular_speed = np.linalg.norm(omc_w, axis=1)
    reliable = valid & good_omc & (angular_speed < 4.5)
    calibration = reliable & (tw >= 5.) & (tw < 120.)
    heldout = reliable & (tw >= 120.) & (tw <= 600.)
    if calibration.sum() < 1000 or heldout.sum() < 1000:
        raise ValueError('Too few reliable samples in the calibration or held-out validation set')
    raw_cal = imuint(query[calibration], ti, gyro)
    ref_cal = omc_w[calibration]
    candidates = []
    for sign in (+1, -1):
        polarity = sign * C
        x = raw_cal @ polarity.T
        xbar, ybar = x.mean(axis=0), ref_cal.mean(axis=0)
        # Kabsch/Procrustes on mean-centered vectors: enforce det(Q)=+1.
        rotation, _ = Rotation.align_vectors(ref_cal - ybar, x - xbar)
        bias_converted = xbar - rotation.inv().apply(ybar)
        bias_raw = polarity.T @ bias_converted
        mapped = (raw_cal - bias_raw) @ (rotation.as_matrix() @ polarity).T
        candidates.append({'sign': sign, 'Q': rotation.as_matrix(), 'bias_raw': bias_raw,
                           'training_RMSE_rad_s': vector_rmse(mapped, ref_cal)})
    candidate = min(candidates, key=lambda r: r['training_RMSE_rad_s'])
    sign, Q, bias = candidate['sign'], candidate['Q'], candidate['bias_raw']
    M_g = Q @ (sign * C)
    M_a = Q @ C
    w_ol = (gyro - bias) @ M_g.T
    f_ol = accel @ M_a.T

    def metrics(x, y):
        err = x - y
        moving = (np.linalg.norm(y, axis=1) > .5) & (np.linalg.norm(x, axis=1) > .05)
        cos = np.sum(x[moving] * y[moving], axis=1) / (np.linalg.norm(x[moving], axis=1) * np.linalg.norm(y[moving], axis=1))
        return {'vector_RMSE_deg_s': vector_rmse(x, y) * 180 / np.pi,
                'median_direction_error_deg': float(np.median(np.degrees(np.arccos(np.clip(cos, -1., 1.))))),
                'OMC_vector_RMS_deg_s': float(np.sqrt(np.mean(np.sum(y**2, axis=1))) * 180 / np.pi),
                'sample_count': int(len(y))}

    raw_test = imuint(query[heldout], ti, gyro)
    ref_test = omc_w[heldout]
    fitted_train = (raw_cal - bias) @ M_g.T
    fitted_test = (raw_test - bias) @ M_g.T
    baseline_test = (raw_test - bias_static) @ (q_tilt.as_matrix() @ C).T
    mapped_accel_up = M_a @ f_static
    mapped_accel_up /= np.linalg.norm(mapped_accel_up)
    gravity_error = float(np.degrees(np.arccos(np.clip(mapped_accel_up @ target_up, -1., 1.))))
    np.savetxt(out/'eun_to_enu_C.csv', C, delimiter=',', fmt='%.10f')
    np.savetxt(out/'converted_to_ol_Q.csv', Q, delimiter=',', fmt='%.10f')
    np.savetxt(out/'gyro_raw_to_ol_M.csv', M_g, delimiter=',', fmt='%.10f')
    np.savetxt(out/'accel_raw_to_ol_M.csv', M_a, delimiter=',', fmt='%.10f')
    csv_xyz(out/'imu_aligned.csv', ti, gyro_ol_rad_s=w_ol, specific_force_ol_m_s2=f_ol)
    pd.DataFrame([{'gyro_channel_sign': c['sign'], 'training_RMSE_deg_s': c['training_RMSE_rad_s']*180/np.pi}
                  for c in candidates]).to_csv(out/'gyro_sign_hypotheses.csv',index=False)
    # The following figure is held-out motion, not calibration data.
    show = heldout & (tw >= 280.) & (tw < 295.)
    xshow = imuint(query[show], ti, gyro)
    yshow = omc_w[show]
    fitshow = (xshow - bias) @ M_g.T
    oldshow = (xshow - bias_static) @ (q_tilt.as_matrix() @ C).T
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(tw[show], yshow[:,j]*180/np.pi, label='OMC',color='black',lw=1.1)
        ax.plot(tw[show], oldshow[:,j]*180/np.pi, label='Gravity-only',color='#999999',lw=.8)
        ax.plot(tw[show], fitshow[:,j]*180/np.pi, label='Dynamic SVD + bias',color='#ce6b35',lw=.85)
        ax.set_ylabel('XYZ'[j]+' (deg/s)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=3)
    axes[-1].set_xlabel('OMC time (s)')
    fig.tight_layout();fig.savefig(out/'gyro_vector_comparison.png',dpi=155);plt.close(fig)
    result = {
        'calibration_method': 'Mean-centered 3-axis Kabsch (proper Q), constant gyro bias from corresponding means, after scalar-norm Task3 synchronization',
        'calibration_interval_omc_s': [5,120], 'validation_interval_omc_s': [120,600],
        'calibration_rules': 'OMC marker fit residual <5 mm; OMC angular speed <4.5 rad/s',
        'recorded_gyro_sign': sign, 'C': C.tolist(), 'Q': Q.tolist(), 'M_gyro': M_g.tolist(), 'M_accel': M_a.tolist(),
        'det_C': float(np.linalg.det(C)), 'det_Q': float(np.linalg.det(Q)),
        'det_M_gyro': float(np.linalg.det(M_g)), 'det_M_accel': float(np.linalg.det(M_a)),
        'gyro_bias_raw_fitted_deg_s': (bias*180/np.pi).tolist(),
        'gyro_bias_raw_static_deg_s': (bias_static*180/np.pi).tolist(),
        'training': metrics(fitted_train, ref_cal),
        'heldout_validation': metrics(fitted_test, ref_test),
        'gravity_only_heldout_baseline': metrics(baseline_test, ref_test),
        'static_accel_gravity_direction_error_deg': gravity_error,
        'static_specific_force_norm_m_s2': float(np.linalg.norm(f_static)),
        'photographs': 'Used to document marker numbering and physical ankle mounting only; not to infer electrical axes',
        'axis_sign_limit': 'Gyro sign candidate chosen on training data; accel and gyro physical parity conventions are not verified by firmware'}
    return w_ol, f_ol, w_baseline, result


def vector_rmse(x, y):
    return float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))


def imuint(tquery, tsource, values):
    return np.column_stack([np.interp(tquery, tsource, values[:, k]) for k in range(values.shape[1])])


def task3(t, p, template, ti, gyro_raw, static_bias_raw):
    out=OUTPUT;out.mkdir(parents=True, exist_ok=True)
    smooth=gaussian_filter1d(median_filter(p, size=(5,1,1)), 3, axis=0)
    r, pos, residual=rigid_fit(smooth,template)
    tw=(t[:-1]+t[1:]) / 2
    # Body angular velocity: Log( R_k^T R_k+1 ) / dt, resolved in OL.
    omc_w=Rotation.from_matrix(np.transpose(r[:-1],(0,2,1))@r[1:]).as_rotvec()/np.diff(t)[:,None]
    omc_speed=gaussian_filter1d(np.linalg.norm(omc_w,axis=1),5)
    imu_speed=np.linalg.norm(gyro_raw-static_bias_raw,axis=1)
    # Clock synchronization uses rotation-invariant magnitudes of raw
    # stationary-bias-removed channels. No fitted Task 2 Q is used here.
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
    csv_xyz(out/'imu_ang_vel_preliminary.csv',alpha*ti+beta,gyro_raw_minus_static_bias_rad_s=gyro_raw-static_bias_raw)
    sel=(tw>=280)&(tw<291)
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
    axes[0].plot(tw[sel],omc_speed[sel],label='OMC')
    axes[0].plot(tw[sel],np.interp(tw[sel],ti,imu_speed),label='IMU no correction')
    axes[1].plot(tw[sel],omc_speed[sel],label='OMC')
    axes[1].plot(tw[sel],np.interp((tw[sel]-beta)/alpha,ti,imu_speed),label='IMU synchronized')
    for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set_ylabel('Angular speed (rad/s)')
    axes[1].set_xlabel('OMC time (s)');fig.tight_layout();fig.savefig(out/'synchronization.png',dpi=150);plt.close(fig)
    return tw,omc_w,r,pos,residual,alpha,beta,{
        'clock_model':'t_OMC = alpha * (t_IMU - t_IMU_first) + beta',
        'alpha':float(alpha),'beta_s':float(beta),'angular_speed_correlation':float(-opt.fun),
        'axis_rotation_independent':True,
        'note':'Clock fitted only from angular-speed magnitudes before any dynamic fixed-axis calibration.'}


def task4(t, w, omc_r, baseline_w=None):
    out=OUTPUT;out.mkdir(parents=True, exist_ok=True)
    # Active quaternion represents OL->OG, scalar-last [qx,qy,qz,qw].
    # Right multiply body-frame increment. Initialize using first OMC pose once.
    def integrate(omega):
        n=len(t);q=np.empty((n,4))
        ori=Rotation.from_matrix(omc_r[0]);q[0]=ori.as_quat()
        for k in range(1,n):
            ori=ori*Rotation.from_rotvec((omega[k-1]+omega[k])*.5*(t[k]-t[k-1]))
            q[k]=ori.as_quat()
        measured_rotation=Rotation.from_matrix(omc_r)
        error=np.degrees((measured_rotation.inv()*Rotation.from_quat(q)).magnitude())
        for k in range(1,len(q)):
            if np.dot(q[k],q[k-1])<0:q[k]*=-1.
        return q,error
    estimated,angle=integrate(w)
    measured=Rotation.from_matrix(omc_r).as_quat()
    for k in range(1,len(measured)):
        if np.dot(measured[k],measured[k-1])<0:measured[k]*=-1.
    baseline_angle=integrate(baseline_w)[1] if baseline_w is not None else None
    csv_xyz(out/'imu_orientation.csv',t,quat_xyzw=estimated)
    csv_xyz(out/'omc_orientation.csv',t,quat_xyzw=measured)
    csv_xyz(out/'orientation_error.csv',t,angle_deg=angle)
    if baseline_angle is not None:
        csv_xyz(out/'gravity_only_baseline_error.csv',t,angle_deg=baseline_angle)
    fig,axs=plt.subplots(2,1,figsize=(10,6))
    axs[0].plot(t,angle,color='#ae4938',lw=.8,label='Dynamic SVD and fitted bias')
    if baseline_angle is not None:
        axs[0].plot(t,baseline_angle,color='#999999',lw=.7,label='Static gravity-only')
    axs[0].legend(fontsize=8)
    initial=t<=t[0]+60.
    axs[1].plot(t[initial],angle[initial],color='#ae4938',lw=.8)
    if baseline_angle is not None:
        axs[1].plot(t[initial],baseline_angle[initial],color='#999999',lw=.7)
    for ax in axs:
        ax.set(xlabel='OMC time (s)',ylabel='Relative rotation error (deg)')
        ax.grid(alpha=.2)
    axs[1].set_title('First 60 s')
    fig.tight_layout();fig.savefig(out/'orientation_error.png',dpi=150);plt.close(fig)
    def summarize(error):
        if error is None:return None
        return {'mean_error_deg':float(error.mean()),
                'final_error_deg':float(error[-1]),
                'max_error_deg':float(error.max()),
                'first_30_s_mean_error_deg':float(error[t<t[0]+30].mean())}
    return Rotation.from_quat(estimated).as_matrix(),{
        'quaternion_order':'[qx,qy,qz,qw], active local OL -> global OG',
        'initial_condition':'OMC first synchronized orientation at overlap start',
        'integration':'q[k+1]=q[k] * exp(0.5*(omega[k]+omega[k+1])*dt), body-frame right product',
        'aligned':summarize(angle),
        'gravity_only_baseline':summarize(baseline_angle)}


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


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    t,p,ti,gyro,accel=extract_raw()
    # Task 1 is the user's completed prior step. Load its saved reference
    # geometry as a fixed read-only input; do not rebuild or overwrite Task 1.
    if not np.isfinite(p).all():
        raise ValueError('Provide completed Task 1 refined OMC data; source export has missing markers.')
    b0=np.loadtxt(INPUT/'task1_initial_OL_to_OG.csv',delimiter=',')
    template=np.loadtxt(INPUT/'task1_template_mm.csv',delimiter=',')
    if b0.shape!=(3,3) or template.shape!=(4,3):
        raise ValueError('Saved Task 1 reference geometry has unexpected dimensions')
    bias_static=gyro[(ti>=1)&(ti<5)].mean(axis=0)
    tw,omc_w,r,pos,residual,alpha,beta,s3=task3(t,p,template,ti,gyro,bias_static)
    w,f,wbase,s2=task2(ti,gyro,accel,b0,alpha,beta,tw,omc_w,residual)
    mask=(t>=max(t[0],alpha*ti[0]+beta))&(t<=min(t[-1],alpha*ti[-1]+beta))
    ts=t[mask];query=(ts-beta)/alpha
    ws=imuint(query,ti,w)
    fs=imuint(query,ti,f)
    wb=imuint(query,ti,wbase)
    rs=r[mask];ps=pos[mask]
    csv_xyz(OUTPUT/'imu_synched.csv',ts,gyro_ol_rad_s=ws,specific_force_ol_m_s2=fs)
    csv_xyz(OUTPUT/'omc_synched.csv',ts,centroid_og_mm=ps,rigid_fit_residual_mm=residual[mask])
    csv_xyz(OUTPUT/'imu_ang_vel_calibrated.csv',alpha*ti+beta,omega_ol_rad_s=w)
    s3['overlap_s']=[float(ts[0]),float(ts[-1])]
    rimu,s4=task4(ts,ws,rs,baseline_w=wb)
    s5=task5(ts,fs,rimu,ps)
    summary={'metadata':{'omc_frames':len(t),'omc_duration_s':float(t[-1]),
                         'imu_samples':len(ti),'imu_duration_s':float(ti[-1]),
                         'omc_frame':'right-handed global ENU, mm',
                         'imu_frame':'recorded left-handed local EUN, deg/s & g'},
             'task2':s2,'task3':s3,'task4':s4,'task5':s5,
             'limitations':['Task 1 is an existing read-only input. This code has no Task 1 calibration or reconstruction steps.',
                            'The computational order revisits Task 2 after Task 3: basic conversions then independent norm-only clock fit, then fixed 3-axis spatial calibration. This is documented rather than obscured.',
                            'Calibration and gyro bias are fitted on 5-120 s; 120-600 s is a held-out validation interval.',
                            'Photographs document mounting location and marker numbering only; no electrical IMU axes inferred from pictures.',
                            'Recorded gyro axial-vector parity/firmware signs and Vicon channel-to-photo marker labels not independently verified.',
                            'Acceleration mapping uses the matched Q and C and is separately gravity-checked; only the gyroscope signs are trained.',
                            'Foot sensor location unknown; OMC marker centroid assumed co-located for Task 5.',
                            'No ZUPT, lever-arm correction, ongoing OMC correction, or heading reference used during IMU integration.',
                            'Rhythm inferred from OMC foot height; recorded verbal cadence unavailable.']}
    (OUTPUT/'results_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ['metadata','task2','task3','task4','task5']},indent=2))


if __name__=='__main__':
    main()
