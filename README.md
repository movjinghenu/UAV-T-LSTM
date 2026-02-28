
# UAV-T-LSTM: ROI-Level UAV–Satellite Fusion for NDVI Time-Series Reconstruction

This repository shares code and ROI-level time-series tables for reconstructing continuous NDVI trajectories from heterogeneous UAV and Sentinel-2 observations under irregular sampling. It also includes multiple baseline reconstructions (linear, spline, T-LSTM, PIGAN, BRIOS) for comparison.

**Repository URL**: https://github.com/movjinghenu/UAV-T-LSTM/

---

## Repository Structure

```

UAV-T-LSTM/
├── code/                    # Python scripts (methods and baselines)
│   ├── UAV-T-LSTM.py        # our method (UAV-T-LSTM / Ours)
│   ├── PIGAN.py             # baseline
│   ├── brios.py             # baseline
│   ├── spline.py            # baseline: spline interpolation
│   └── lin.py               # baseline: linear interpolation
└── data/                    # ROI-level tables (no raw UAV imagery here)
├── figdata.xlsx         # multi-year DOY-aligned NDVI & reconstructions
└── TEM24.xlsx           # 2024 temperature series for event-window analysis

````

---

## What is Included

### Code (`code/`)
- **UAV-T-LSTM** (our reconstruction model)
- Baselines: **Linear**, **Spline**, **T-RNN**, **PIGAN**, **BRIOS**

### Data tables (`data/`)
- `figdata.xlsx`: multi-year ROI-level NDVI tables aligned on **DOY** (day of year), including observation channels and reconstructed trajectories from different methods
- `TEM24.xlsx`: 2024 temperature series used to define an external event window for disturbance analysis

> Note: The raw 2024 UAV multispectral imagery (~1.33GB) is intentionally **not hosted in this GitHub repository**. See **Raw UAV Imagery** below.

---

## Quick Start (Optional)

This section is only needed if you want to execute the scripts locally.  
You may skip it if you only need to browse the code and data tables.

### 1) Install Python dependencies
Recommended: Python 3.9+

```bash
pip install numpy pandas scipy matplotlib openpyxl
````

If a script requires deep learning frameworks (e.g., PyTorch), install them accordingly:

```bash
pip install torch
```

### 2) Run a method script

Example (the exact script behavior depends on each file):

```bash
python code/spline.py
python code/lin.py
python code/PIGAN.py
python code/brios.py
python code/UAV-T-LSTM.py
```

---

## Data Dictionary (Summary)

### `data/figdata.xlsx`

Purpose: DOY-aligned ROI-level NDVI time series and reconstructions from multiple methods across years.

Typical sheets:

* `24data`, `25data`: year 2024/2025 (UAV + satellite observation channels + reconstructions)
* `21data`, `22data`, `23data`: historical years (typically satellite observations + reconstructions)

Common columns (depending on sheet):

* `doy` (int): day of year (1–365/366)
* `s_data` (float): Sentinel-2 ROI NDVI observations (NaN if missing)
* `u_data` (float): UAV ROI NDVI observations (NaN if missing; may exist only for certain years)
* `lin` or `line` (float): linear interpolation reconstruction
* `spline` (float): spline interpolation reconstruction
* `T-LSTM` (float): T-LSTM reconstruction
* `PIGAN` (float): PIGAN reconstruction
* `brios` (float): BRIOS reconstruction
* `net` (float): our method reconstruction (UAV-T-LSTM / Ours)

Some sheets may additionally contain structural metrics (or embedded summaries), such as:

* `CE`: curvature energy
* `TV`: total variation
* `N_peaks`, `N_valleys`, `N_turning_points`: turning-point related counts

### `data/TEM24.xlsx`

Purpose: 2024 temperature series used for defining an external event window in disturbance evaluation.

Sheet:

* `Sheet2`

Columns:

* `DATE` (date): `YYYY-MM-DD`
* `TEM` (float): temperature values (units/definition follow the accompanying manuscript or thesis)

---


## Raw UAV imagery (Zenodo)
- Dataset (v1.0): https://doi.org/10.5281/zenodo.18778904
- File: UAV-T-LSTM-24UAVFIG.7z (~1.4 GB)
- License: CC BY 4.0
---

## License

* Code: see `LICENSE`
* Data tables in `data/`: released for research and academic use. Please cite the associated work if you use these tables.

---

## Citation

If you use this repository in academic work, please cite the associated paper/thesis.

A BibTeX entry will be added after formal publication. Until then, please cite as:

> Zhao, L. (2026). UAV-T-LSTM: ROI-level UAV–Satellite fusion for NDVI time-series reconstruction under irregular sampling. GitHub repository: [https://github.com/movjinghenu/UAV-T-LSTM/](https://github.com/movjinghenu/UAV-T-LSTM/)

---

## Contact

* GitHub Issues: please open an issue in this repository for questions or bug reports.
* Author: Zhao Liang (赵亮)
* Repository: [https://github.com/movjinghenu/UAV-T-LSTM/](https://github.com/movjinghenu/UAV-T-LSTM/)

```
