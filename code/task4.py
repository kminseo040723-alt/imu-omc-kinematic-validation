"""Task 4: quaternion orientation integration and OMC comparison."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from task_common import OUTPUT, csv_xyz


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
