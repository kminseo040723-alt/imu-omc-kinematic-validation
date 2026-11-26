MOBI311 Task 1 – Report-ready 3D figures for Section 3.1
==========================================================

figures/Fig1_Marker_Pair_Geometry_3D.png
  3D spatial companion to the original six-pair distance time series.
  Left: frame at 236.80 s with distances near their medians.
  Right: frame at 360.26 s with the largest summed pair-distance deviation.
  Six pair connections are drawn. Red links differ from pair medians by >5 mm.
  Each frame's 3D marker configuration was orientation-normalized for visual
  comparison using the already saved Task 1 reference. This does not replace
  the original distance-versus-time Figure 1 or its original outlier bands.

figures/Fig2_One_Marker_Rigid_Reconstruction_3D.png
  3D view of an artificial 10-frame gap (297.63–297.72 s) in Marker 1.
  Other three markers are fitted to the saved Task 1 reference with proper
  least-squares rigid-body alignment. Blue = available, gray = withheld
  original, red = reconstructed samples. Right panel enlarges the gap region.

figures/Fig3_Four_Marker_Spline_Reconstruction_3D.png
  The same 10-frame gap is artificially masked for all four markers.
  Each X/Y/Z coordinate is independently reconstructed using SciPy's cubic
  spline on observed samples in 297.30–298.10 s excluding the gap.
  Blue = available, gray = withheld original, red crosses = reconstructed.

IMPORTANT: VALIDATION LIMITATION
--------------------------------
The saved Task 1 marker file and reference were available, but the actual
script that originally produced the reported Table 1 RMSEs 8.48 mm and 0.40
mm for the exact 297.63–297.72-s interval was not available. These 3D figures
are independently reconstructed demonstrations using stated algorithms.
They should not be represented as an exact replay of that missing script.

Independent 3D-vector RMSEs from this script:
  One-marker proper rigid fit: 8.0819 mm (report: 8.48 mm)
  Four-marker SciPy CubicSpline: 1.0253 mm (report: 0.40 mm)

Different template selection, interpolation boundary handling, gap definition
or error metric may explain differences; this has not been established.
Keep your original Table 1 only if its values can be traced to the original
Task 1 code. Do not silently replace or equate the two result sets.

DATA / REPRODUCIBILITY
----------------------
inputs/omc_refined_task1.csv – unmodified copy of Task 1 refined positions.
inputs/empirical_template_mm.csv – unmodified Task 1 reference geometry.
make_task1_report_3d.py – script to regenerate all PNGs and a JSON note.
Run: python make_task1_report_3d.py (requires numpy pandas scipy matplotlib)

SUGGESTED REPORT CAPTIONS
-------------------------
3D companion to Fig. 1. Six inter-marker connections at representative and
anomalous frames. Marker configurations were aligned to the reference pose
for visualization; red connections deviate by more than 5 mm from the
corresponding median inter-marker distance.

Fig. 2 (3D). Rigid-body reconstruction of a synthetic gap in Marker 1 during
297.63–297.72 s. Blue denotes available coordinates, gray the withheld
original, and red reconstructed positions. The right panel enlarges the gap.

Fig. 3 (3D). Cubic-spline reconstruction of simultaneous synthetic gaps in
all four markers during 297.63–297.72 s. Blue denotes available coordinates,
gray the withheld original, and red crosses reconstructed positions.
