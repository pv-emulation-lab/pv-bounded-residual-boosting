# Bounded Residual Boosting for Daily Photovoltaic Energy Emulation

Manuscript source, data, code and results for:

**Bounded Residual Boosting for Daily Photovoltaic Energy Emulation Using Local NASA POWER Data: A Case Study of Cagayan de Oro City, Philippines**

Ar-gie Longganay, Kaiser Cultura (University of Science and Technology of Southern Philippines) and Abid Yahya (Botswana International University of Science and Technology)

> **Scope note.** All photovoltaic outputs in this repository are **simulated** with pvlib for a generic 1-kWp fixed array. They are not measured production. Inputs come from a NASA POWER grid cell at 8.486° N, 124.657° E, not from a campus meter or radiometer.

## What this study does

Daily photovoltaic emulators can look limited by the learning algorithm when their inputs omit the subdaily irradiance information used to generate the reference output. Using 43,824 hourly NASA POWER records (2021–2025), the study compares:

- daily-weather regressors (linear, random forest, XGBoost),
- a plane-of-array (POA) ratio estimator,
- direct XGBoost with matched inputs, and
- the proposed **bounded residual boosting** model: a calibrated POA energy anchor plus an XGBoost correction on daily irradiation, temperature and wind moments.

Hyperparameters are selected on 2024 after fitting on 2021–2023; final models are fitted on 2021–2024 and evaluated on the 365 days of 2025.

### Headline results (2025, simulated reference)

| Model | MAE (kWh/day/kWp) | R² |
|---|---|---|
| Daily-weather XGBoost | 0.26115 | 0.86889 |
| POA ratio estimator | 0.05300 | 0.99436 |
| Direct boosting, matched moment inputs | 0.01961 | 0.99799 |
| **Residual boosting, moment inputs (proposed)** | **0.01049** | **0.99968** |

Adding irradiance-closure discrepancies does not improve the proposed model.

### Limitations

- Retrospective experiments on simulator output; no measured-production validation.
- Architecture development extended an earlier benchmark that already reported 2025 results, so the 2025 evaluation is not a preregistered confirmatory test.
- Historical windows reuse settings selected on 2024 and are not nested prospective tests.
- The DNI intervention changes the reference target; it does not show which irradiance product is more accurate.
- No end-to-end runtime advantage over pvlib is claimed.

## Repository layout

```
.
├── main.tex                      Manuscript source
├── references.bib                Bibliography
├── manuscript-preview.pdf        Compiled manuscript
├── LICENSE                       MIT License (code)
├── figures/                      Seven figures (PDF, SVG, PNG)
│   └── reproducibility/          Data, code and results
│       ├── reference.py          pvlib reference model and daily aggregation
│       ├── analysis.py           Features, tuning, baselines, ablations, intervals
│       ├── plots.py              Generates all manuscript figures
│       ├── requirements.txt      Pinned dependencies
│       ├── SHA256SUMS.txt        Checksums at delivery
│       ├── data/
│       │   ├── nasa-power-hourly.csv        Hourly weather archive
│       │   ├── archived-daily-reference.csv Reference series for checks
│       │   └── raw-responses/               60 original monthly API responses
│       └── results/              Derived features, predictions, tables, metadata
```

## Reproduce the analysis

Requires Python 3.12. No network access is needed; all weather data are included.

```sh
cd figures/reproducibility
python -m pip install -r requirements.txt
python analysis.py
python plots.py
```

These commands overwrite the files in `results/`. Row subsampling uses seed 42, and month-block intervals use 2,000 resampling draws with seed 42. Floating-point results may differ slightly across platforms and dependency builds. Verify file integrity with `SHA256SUMS.txt`.

## Compile the manuscript

From the repository root:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses the PDF figures in `figures/` and does not need the Python package to compile.

## Data provenance

Weather data: NASA POWER (https://power.larc.nasa.gov/), hourly point endpoint, community RE, sources SYN1DEG, MERRA2 and POWER. The archive was downloaded on 31 October 2026, and the original response headers are preserved in `figures/reproducibility/data/raw-responses/`. Please acknowledge NASA POWER when reusing the data.

## License

Code: MIT License (see `LICENSE`). The manuscript text and figures are shared for review and reproducibility; contact the authors before reuse. NASA POWER data, pvlib, XGBoost and other dependencies keep their own licenses and terms.

## Citation

If you use this repository, please cite the manuscript. Journal and DOI details will be added after publication.
