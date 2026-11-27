"""Shared paths, constants, data loading and numerical helpers for Tasks 2--5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

def vector_rmse(x, y):
    return float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))

def imuint(tquery, tsource, values):
    return np.column_stack([np.interp(tquery, tsource, values[:, k]) for k in range(values.shape[1])])
