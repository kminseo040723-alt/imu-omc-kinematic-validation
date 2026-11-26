MOBI311 TASKS 2-5 RECALCULATION
===============================

Folders
-------
inputs/  : Read-only copies of the user's existing raw IMU file and already
           completed Task 1 OMC/ref-frame outputs. Task 1 is not recomputed.
code/    : run_tasks2to5.py, requirements.txt
results/ : Computed numerical files and plots in task2/, task3/, task4/, task5/
report/  : MOBI311_Tasks2to5_DynamicCalibration.docx

Reproduce
---------
1. Install the packages in code/requirements.txt.
2. Run: python code/run_tasks2to5.py
3. Inspect results/results_summary.json.

Actual numerical execution sequence
-----------------------------------
Preliminary Task 2: EUN-to-ENU channel permutation, units, static gyro bias.
Task 3: clock synchronization from angular-speed magnitudes (no Q fitted).
Final Task 2: fixed 3-axis mapping and constant gyro bias from synchronized
              5-120 s OMC/IMU angular velocities by a proper SVD rotation.
Validation:   120-600 s held-out IMU/OMC gyro vector mismatch.
Task 4:       time-varying quaternion integration initialized at first OMC pose.
Task 5(a):    required three 4th-order Butterworth band-pass filters.
Task 5(b):    OMC-foot-period-informed band on the same 30-second segment.

The analytical order revisits Task 2 after Task 3. This is explicitly
documented: a scalar angular-speed norm is invariant under the later
fixed orthogonal spatial mapping, so the clock fit has no Q dependency.
It should not be described as a strictly one-pass sequential algorithm.

Calibration is evaluated on a DISJOINT 120-600 s interval. The photograph
documents the numbered markers and module mounting, not sensor electrical
axes. Channel signs and accelerometer-gyroscope axis compatibility were
checked operationally, not independently verified from firmware.

Task 5 compares against the four-marker CENTROID: the actual accelerometer
location is not precisely known, and an exact lever-arm correction was not
possible. No ZUPT, feedback from OMC, or hidden drift resets were used.
The 10-minute integrated trajectory therefore remains inaccurate.
