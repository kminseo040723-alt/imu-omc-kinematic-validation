#!/usr/bin/env python3
"""Task 1: Vicon marker diagnostics, conservative gap filling, held-out validation.

Input: Vicon Trajectories CSV (as exported), or flat omc_time + marker_1_X...4_Z.
Coordinates stay in mm and their original global frame. No IMU processing here.
Run: python task1_marker_refinement.py --input "record4(1).csv" --demo
"""
from pathlib import Path
import argparse
import csv
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

PAIRS = list(combinations(range(4), 2))
COLS = [f'marker_{m}_{axis}' for m in range(1, 5) for axis in 'XYZ']


def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip() == 'Trajectories':
        rate = float(rows[1][0])
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid Vicon sample rate.')
        frame = next(i for i, r in enumerate(rows) if r and r[0].strip() == 'Frame')
        names = [v.strip() for v in rows[frame - 1][2:] if v.strip()]
        labels = rows[frame][2:]
        if labels != ['X', 'Y', 'Z'] * 4:
            raise ValueError('Expected exactly four markers with XYZ columns.')
        units = rows[frame + 1][2:14]
        if any(v.strip() != 'mm' for v in units):
            raise ValueError('This script requires Vicon coordinates in mm.')
        numeric_rows = [r for r in rows[frame + 2:] if any(v.strip() for v in r)]
        data = pd.DataFrame(numeric_rows).iloc[:, :14]
        data = data.apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if data.shape[1] != 14 or not np.isfinite(data[:, :2]).all():
            raise ValueError('Invalid Frame/Sub Frame columns.')
        if np.any(data[:, 1] != 0):
            raise ValueError('Nonzero Sub Frame: confirm timing convention before use.')
        frames = data[:, 0]
        t = (frames - frames[0]) / rate
        p = data[:, 2:14].reshape(-1, 4, 3)
        meta = dict(format='Vicon Trajectories', source_sample_rate_hz=rate,
                    first_frame=float(frames[0]), marker_labels_in_column_order=names)
    else:
        df = pd.read_csv(path)
        if 'omc_time' not in df or not all(c in df for c in COLS):
            raise ValueError('Need Vicon Trajectories format or omc_time, marker_1_X...marker_4_Z.')
        t = df.omc_time.to_numpy(float)
        p = df[COLS].to_numpy(float).reshape(-1, 4, 3)
        frames = np.arange(len(t))
        meta = dict(format='flat CSV', marker_labels_in_column_order=list(range(1, 5)))
    if len(t) < 10 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Need at least 10 frames with strictly increasing finite time.')
    # A partial XYZ observation is unusable; zero XYZ is NOT missing by default.
    p[~np.isfinite(p).all(axis=2)] = np.nan
    return t, frames, p, meta


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def diagnostics(p, floor_mm, mad_scale):
    d = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in PAIRS], axis=1)
    if any(np.isfinite(d[:, j]).sum() < 10 for j in range(6)):
        raise ValueError('Insufficient pair distances for empirical geometry. Supply more usable data.')
    med = np.nanmedian(d, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(d - med), axis=0)
    tol = np.maximum(floor_mm, mad_scale * mad)
    bad = np.isfinite(d) & (np.abs(d - med) > tol)
    # Attribute a suspect marker only if its three edges fail and the remaining
    # triangle is fully observed and passes. Ambiguous bad frames are only flagged.
    suspect = np.zeros(p.shape[:2], bool)
    for m in range(4):
        incident = [j for j, pair in enumerate(PAIRS) if m in pair]
        opposite = [j for j, pair in enumerate(PAIRS) if m not in pair]
        suspect[:, m] = (bad[:, incident].all(axis=1)
                         & (~bad[:, opposite]).all(axis=1)
                         & np.isfinite(d[:, opposite]).all(axis=1))
    return d, med, tol, bad, suspect


def fit_pose(local, observed):
    """Row-vector Kabsch: global = local @ R + translation; det(R) = +1."""
    a = local - local.mean(axis=0)
    b = observed - observed.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-6) < 2 or np.linalg.matrix_rank(b, tol=1e-6) < 2:
        raise ValueError('Collinear marker anchors.')
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    r = u @ correction @ vt
    translation = observed.mean(axis=0) - local.mean(axis=0) @ r
    return r, translation


def choose_template(p, distances, med, tol, bad):
    complete = np.isfinite(p).all(axis=(1, 2)) & ~bad.any(axis=1)
    candidates = np.flatnonzero(complete)
    if not len(candidates):
        raise ValueError('No complete, geometrically consistent frame for empirical template.')
    score = np.mean(((distances[candidates] - med) / tol) ** 2, axis=1)
    idx = int(candidates[np.argmin(score)])
    # The first marker is the template origin, but this is NOT the assignment OL calibration.
    return p[idx] - p[idx, 0], idx


def refine(t, p, template, tol, max_gap, rigid_limit):
    out = p.copy()
    available = np.isfinite(p).all(axis=2)
    status = np.where(available, 'observed', 'unresolved').astype('<U16')
    # Only ORIGINAL available coordinates serve as anchors. No recursive confidence inflation.
    for k in np.flatnonzero((available.sum(axis=1) == 3)):
        seen = np.flatnonzero(available[k])
        if any(abs(np.linalg.norm(p[k, i] - p[k, j]) - np.linalg.norm(template[i] - template[j]))
               > tol[PAIRS.index((i, j))] for i, j in combinations(seen, 2)):
            continue
        try:
            r, tr = fit_pose(template[seen], p[k, seen])
        except ValueError:
            continue
        fit = template[seen] @ r + tr
        if np.sqrt(np.mean(np.sum((fit - p[k, seen]) ** 2, axis=1))) > rigid_limit:
            continue
        missing = np.flatnonzero(~available[k])
        out[k, missing] = template[missing] @ r + tr
        status[k, missing] = 'rigid'
    # With <=2 markers, instantaneous orientation is underdetermined. Interpolate
    # only short, internal gaps; never extrapolate. Use original observations only.
    for m in range(4):
        for start, end in runs(status[:, m] == 'unresolved'):
            if start == 0 or end == len(t) or t[end] - t[start - 1] > max_gap:
                continue
            left = np.flatnonzero(available[:start, m])[-2:]
            right = np.flatnonzero(available[end:, m])[:2] + end
            if not len(left) or not len(right):
                continue
            # Require adjacent boundary observations; avoid bridging an earlier unresolved gap.
            if left[-1] != start - 1 or right[0] != end:
                continue
            anchors = np.r_[left, right]
            method = 'spline' if len(left) == 2 and len(right) == 2 else 'linear'
            for axis in range(3):
                if method == 'spline':
                    out[start:end, m, axis] = CubicSpline(t[anchors], p[anchors, m, axis],
                                                       bc_type='natural')(t[start:end])
                else:
                    out[start:end, m, axis] = np.interp(t[start:end], t[anchors], p[anchors, m, axis])
            status[start:end, m] = method
    return out, status


def save_coordinates(path, t, frames, p, status):
    df = pd.DataFrame(p.reshape(-1, 12), columns=COLS)
    df.insert(0, 'omc_time', t)
    df.insert(0, 'source_frame', frames)
    for m in range(4):
        df[f'marker_{m+1}_status'] = status[:, m]
    df.to_csv(path, index=False)


def plot_gap(path, t, truth, masked, result, status, m, start, end, title):
    lo, hi = max(0, start - 35), min(len(t), end + 35)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for a, axis in enumerate(axes):
        axis.plot(t[lo:hi], truth[lo:hi, m, a], color='0.65', label='Input / held-out reference')
        axis.plot(t[lo:hi], masked[lo:hi, m, a], 'b-', lw=1.2, label='Available observations')
        restored = np.isin(status[lo:hi, m], ['rigid', 'spline', 'linear'])
        axis.plot(t[lo:hi][restored], result[lo:hi, m, a][restored], 'r.', ms=4, label='Reconstructed')
        axis.axvspan(t[start], t[end - 1], color='orange', alpha=.12)
        axis.set_ylabel('XYZ'[a] + ' (mm)')
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel('Time from exported first frame (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heldout_demo(outdir, t, frames, p, d, med, tol, bad, args):
    gap_n = max(2, int(round(args.demo_gap * (1 / np.median(np.diff(t))))))
    if gap_n + 4 >= len(t):
        raise ValueError('Demo gap too long.')
    good = np.isfinite(p).all(axis=(1, 2)) & ~bad.any(axis=1)
    # Need clean context around the gap. Deterministic selection; not selected by reconstruction RMSE.
    candidates = [s for s in range(2, len(t) - gap_n - 2)
                  if good[s-2:s+gap_n+2].all()]
    if not candidates:
        return dict(skipped='No clean interval for held-out validation.')
    start = candidates[len(candidates) // 2]
    end = start + gap_n
    report = []
    for name, markers in [('one_marker', [0]), ('all_markers', [0, 1, 2, 3])]:
        masked = p.copy()
        masked[start:end, markers, :] = np.nan
        # Re-estimate all calibration statistics WITHOUT held-out samples.
        dd, mm, tt, bb, _ = diagnostics(masked, args.distance_floor_mm, args.mad_scale)
        template, template_idx = choose_template(masked, dd, mm, tt, bb)
        restored, status = refine(t, masked, template, tt, args.max_gap, args.rigid_fit_mm)
        reference = p[start:end][:, markers]
        estimate = restored[start:end][:, markers]
        valid = np.isfinite(estimate).all(axis=2)
        errors = np.linalg.norm(estimate - reference, axis=2)[valid]
        rec = dict(case=name, artificial_gap=True, start_s=float(t[start]), end_s=float(t[end-1]),
                   samples_per_marker=gap_n, tested_markers=[m+1 for m in markers],
                   template_frame=int(template_idx), coverage=float(valid.mean()),
                   position_rmse_mm=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                   max_position_error_mm=float(errors.max()) if len(errors) else None,
                   methods=sorted(set(status[start:end][:, markers].ravel())))
        report.append(rec)
        save_coordinates(outdir / f'task1_demo_{name}.csv', t, frames, restored, status)
        plot_gap(outdir / f'task1_demo_{name}.png', t, p, masked, restored, status, markers[0],
                 start, end, 'ARTIFICIAL gap validation: ' + name)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'raw_csv' / 'omc_vicon.csv')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    ap.add_argument('--distance-floor-mm', type=float, default=5.0,
                    help='Minimum pair-distance flag threshold; diagnostic choice, not sensor accuracy.')
    ap.add_argument('--mad-scale', type=float, default=6.0)
    ap.add_argument('--rigid-fit-mm', type=float, default=3.0)
    ap.add_argument('--max-gap', type=float, default=.20, help='Max boundary-to-boundary interpolation time (s).')
    ap.add_argument('--zero-is-missing', action='store_true', help='Only use if zero XYZ denotes missing in your export.')
    ap.add_argument('--repair-suspects', action='store_true', help='Opt-in: mask conservative geometry suspects before reconstruction.')
    ap.add_argument('--demo', action='store_true', help='Also run separate, explicitly artificial held-out tests.')
    ap.add_argument('--demo-gap', type=float, default=.10)
    args = ap.parse_args()
    if min(args.distance_floor_mm, args.mad_scale, args.rigid_fit_mm, args.max_gap, args.demo_gap) <= 0:
        ap.error('All threshold/duration values must be positive.')
    args.out.mkdir(parents=True, exist_ok=True)
    t, frames, original, meta = load_csv(args.input)
    p = original.copy()
    if args.zero_is_missing:
        p[np.all(p == 0, axis=2)] = np.nan
    d, med, tol, bad, suspect = diagnostics(p, args.distance_floor_mm, args.mad_scale)
    template, template_idx = choose_template(p, d, med, tol, bad)
    source = p.copy()
    if args.repair_suspects:
        source[suspect] = np.nan
    refined, status = refine(t, source, template, tol, args.max_gap, args.rigid_fit_mm)
    save_coordinates(args.out / 'task1_omc_refined.csv', t, frames, refined, status)
    flags = pd.DataFrame({'omc_time': t, 'source_frame': frames,
                          'geometry_flag': bad.any(axis=1)})
    pair_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for j, (i, k) in enumerate(PAIRS):
        tag = f'{i+1}_{k+1}'
        flags[f'distance_{tag}_mm'] = d[:, j]
        flags[f'pair_{tag}_flag'] = bad[:, j]
        pair_rows.append(dict(pair=tag, median_mm=float(med[j]), threshold_mm=float(tol[j]),
                              min_mm=float(np.nanmin(d[:, j])), max_mm=float(np.nanmax(d[:, j])),
                              flagged_frames=int(bad[:, j].sum())))
        axis = axes.ravel()[j]
        axis.plot(t, d[:, j], lw=.4)
        axis.axhline(med[j], color='k', lw=.7)
        axis.axhspan(med[j]-tol[j], med[j]+tol[j], color='green', alpha=.12)
        axis.plot(t[bad[:, j]], d[bad[:, j], j], 'r.', ms=1)
        axis.set_title(f'Marker {i+1}-{k+1}')
        axis.set_ylabel('Distance (mm)')
        axis.set_xlabel('Time (s)')
    fig.suptitle('Pair-distance diagnostics: red = flagged, not confirmed error')
    fig.tight_layout()
    fig.savefig(args.out / 'task1_pair_distances.png', dpi=160)
    plt.close(fig)
    for m in range(4):
        flags[f'marker_{m+1}_suspect'] = suspect[:, m]
        flags[f'marker_{m+1}_input_missing'] = ~np.isfinite(p[:, m]).all(axis=1)
    flags.to_csv(args.out / 'task1_quality_flags.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(args.out / 'task1_pair_distance_summary.csv', index=False)
    np.savetxt(args.out / 'task1_empirical_template_mm.csv', template, delimiter=',', header='X_mm,Y_mm,Z_mm')
    # Actual gaps and validation outputs remain distinct.
    plotted = 0
    for m in range(4):
        for start, end in runs(~np.isfinite(source[:, m]).all(axis=1)):
            if plotted >= 4:
                break
            plot_gap(args.out / f'task1_actual_gap_{plotted+1}.png', t, p, source, refined, status,
                     m, start, end, f'Actual input/masked gap: marker {m+1}')
            plotted += 1
    summary = dict(input=str(args.input), units='mm', metadata=meta, samples=len(t),
                   duration_s=float(t[-1]-t[0]), empirical_template_frame_index=template_idx,
                   template_note='One consistent observed frame; not supplied assignment geometry or IMU calibration.',
                   missing_marker_frames=int((~np.isfinite(p).all(axis=2)).sum()),
                   flagged_frames=int(bad.any(axis=1).sum()), suspect_marker_frames=int(suspect.sum()),
                   repair_suspects_enabled=args.repair_suspects,
                   statuses={str(s):int((status == s).sum()) for s in np.unique(status)},
                   parameters={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   limitations=['Geometry flags are heuristic, not proof of an error.',
                                'Existing Vicon-filled intervals cannot be identified from coordinates alone.',
                                'Interpolation is limited to short internal gaps; long/edge gaps stay unresolved.',
                                'Held-out error measures agreement with exported data, not absolute accuracy.',
                                'Rigid-body assumptions may fail if the marker mounting deforms.',
                                'A single-frame empirical template can retain measurement error.'])
    if args.demo:
        summary['heldout_validation'] = heldout_demo(args.out, t, frames, p, d, med, tol, bad, args)
    with open(args.out / 'task1_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Saved:', args.out.resolve())


if __name__ == '__main__':
    main()
