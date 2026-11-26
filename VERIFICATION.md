# Reproduction check — 2026-11-26

The recovered source files were retained without algorithm changes.

- Parsed all extracted Python source files successfully.
- Re-ran `analysis/dynamic_calibration/code/run_tasks2to5.py` using a separate copy of its recovered inputs and code, preserving archived results.
- Compared all 115 numeric leaves in the resulting `results_summary.json` against the original summary using `numpy.isclose(rtol=1e-5, atol=1e-6)`: **115 / 115 passed**.
- Environment: Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; Matplotlib 3.10.8.
- Acquisition hardware was not connected or retested. Earlier alternative pipelines and the separate Week 6 packages were preserved but not rerun in this check.

This establishes numerical reproduction of the saved primary analysis in this environment, not independent validation of its physical assumptions or the original measurement accuracy.
