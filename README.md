# DA Comparison Utilities

Quick scripts for photometry QC and plotting.

## Contents
- `Test_doric.py`: QC for Doric `.doric` recordings (single ROI). Loads ROI signals, aligns channels, computes corrected dF/F, z-scores, and saves a 3-panel PDF per file.
- `doric.py`: Minimal helper to read Doric HDF5 files (`h5read`, structure inspection).
- `Test_TDT.ipynb`: TDT photometry dF/F loading/plotting (raw 405/465 + dF/F). Reads RAW `.mat/.h5`, dFF `.mat`, aligns to timebase, and makes combined plots.

## Requirements
- Python 3.x (Anaconda recommended)
- Packages: `numpy`, `pandas`, `h5py`, `matplotlib`, `scipy`
- For TDT: `tdt` package (if reading TDT blocks directly)
- For Doric `.doric`: Doric photometry reader (e.g., `doric` from ethierlab) or local `doric.py` helper.

## Usage

### Doric QC
1) Set `DIR_PATH` in `Test_doric.py` to the folder with `.doric` files.
2) Run:
```
C:\Users\kouhi\anaconda3\python.exe Test_doric.py
```
3) PDFs (`*_quickQC_singleROI.pdf`) are saved next to the data.

### TDT dF/F plotting
1) Set `dFF_file_path` and `RAW_file_path` in `Test_TDT.ipynb` (or the script).
2) Set `box` index.
3) Run the notebook/script. It:
   - Loads dF/F (`dFF`/`dFFOut` MAT)
   - Loads RAW timebase/events
   - Finds raw 405/465 in the RAW file, plots raw + dF/F in one figure
   - Optionally saves the figure

## Notes
- If `import doric` fails, install the Doric photometry reader from the GitHub source (not the PyPI ML package) or use the included `doric.py`.
- If raw 405/465 aren�t auto-detected in the RAW file, inspect printed dataset names and set `cand_405`/`cand_465` accordingly.
- Sampling rate is inferred from dF/F length and time vector; adjust if your files differ.
