"""Task 2: fixed spatial calibration and constant gyroscope bias.

The task2 function is called by the existing 'task 2-5' runner after clock synchronization.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from task_common import C, OUTPUT, csv_xyz, imuint, vector_rmse


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
