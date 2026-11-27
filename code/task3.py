"""Task 3: OMC angular velocity and rotation-invariant clock synchronization."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from task_common import OUTPUT, csv_xyz


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
