MOBI311 – Task 1 Results in 3D
================================

FILES
-----
visuals/Task1_3D_Overview.png
  Static 3D figure for insertion into a report.
  Panel A: four global trajectories and their centroid in meters.
  Panel B: the saved four-marker empirical reference in the Task 1 OL frame,
           centered at the marker centroid and expressed in millimeters.

visuals/Task1_3D_Interactive.html
  Standalone interactive HTML. Double-click to open in a desktop browser.
  Drag either 3D canvas to rotate. Use the mouse wheel to zoom, and the time
  slider or Play button to inspect marker motion. Left shows global ENU
  motion; right shows the marker cluster relative to its instantaneous
  centroid in global axis directions. Its interactive preview uses 0.5-second
  samples for responsiveness, while the original dataset remains at 100 Hz.
  The HTML needs no internet connection, installation, or external scripts.

create_task1_3d.py
  Source code to regenerate both visualization files.

inputs/omc_refined_task1.csv
  Unmodified copy of the already completed Task 1 marker positions.

inputs/empirical_template_mm.csv
  Unmodified copy of the previously completed Task 1 local reference geometry.

METHOD / DATA
-------------
60,806 frames, 0–608.05 seconds, 100 Hz, four numbered marker channels.
No Task 1 reconstruction, marker re-numbering, or spatial calibration was
recomputed.  The plot adopts the existing mapping of marker_1...marker_4 in
the saved Task 1 files; photo-to-export channel identity remains an assumption.
The local reference is shown in millimeters and the global motion in meters.

REGENERATE
----------
Install Python packages: numpy pandas matplotlib
From this directory: python create_task1_3d.py
