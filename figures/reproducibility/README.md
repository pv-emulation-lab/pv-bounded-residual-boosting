# Daily photovoltaic emulation: reproducibility package

This package supports **Bounded Residual Boosting for Daily Photovoltaic Energy Emulation Using Local NASA POWER Data: A Case Study of Cagayan de Oro City, Philippines**.

The response is a deterministic pvlib simulation, not measured photovoltaic production. Inputs describe the NASA POWER grid cell at 8.486 N, 124.657 E. The electrical system is a generic assumed 1-kWp fixed array.

## Manuscript compilation

Run these commands from the repository root, where `main.tex`, `references.bib` and the `figures` directory are located. Compile with:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The PDF preview is supplied in the figures archive. All seven figures are available as vector PDF, editable SVG and 320-dpi PNG. The manuscript references the PDFs. It does not require the Python package to compile.

## Reproduce the analysis

Use Python 3.12 and install the versions in `requirements.txt`. From the `figures/reproducibility` directory:

```sh
python -m pip install -r requirements.txt
python analysis.py
python plots.py
```

The archive contains all required weather data. No network call is required for reproduction. Results are overwritten by these commands. Row subsampling uses seed 42; month-block intervals use 2,000 common resampling draws with seed 42. Floating-point results may vary slightly with platform or dependency builds.

- `reference.py`: original physical configuration and daily aggregation, retained with a corrected relative data path.
- `analysis.py`: input-only optical preprocessing, moment features, chronological tuning, baselines, direct and residual ablations, predictions, intervals, regimes, historical diagnostics and DNI intervention.
- `plots.py`: all manuscript figures. Labels map internal model names to manuscript names.
- `data/nasa-power-hourly.csv`: concatenated original hourly weather archive.
- `data/raw-responses/`: all 60 original monthly API responses, including request strings, source metadata, units and download date (31 October 2026).
- `data/archived-daily-reference.csv`: original reference series for numerical reproduction checks.
- `results/`: derived features, all 365 evaluation predictions, tuning scores, paired comparisons, regimes, historical windows, intervention results and actual environment metadata.
- `SHA256SUMS.txt`: checksums for the reproducibility files at delivery.

The proposed configuration is `Residual moments`. `Expanded moments` is its matched-input direct-XGBoost comparator. `Residual full` and `Expanded XGB` add closure features. Closure features are not used by the selected method. Features are computed without invoking the electrical or thermal reference response in the prediction branch.

## Experimental scope

Architecture development extends an earlier benchmark that already reported 2025 results. Hyperparameters are selected on 2024 without fitting 2025 responses, but the overall extension remains retrospective. Earlier historical windows reuse settings selected with 2024 and are not nested prospective tests. The DNI intervention changes the reference target and is not validation of an improved irradiance product. Projection is inactive for the delivered 2025 residual predictions. No new end-to-end runtime claim is made.

## Repository and release

This package is hosted at https://github.com/ckaisermark/pv-bounded-residual-boosting. A permanent release DOI can be added after archival registration.

The funding, competing-interest and author-contribution statements in `main.tex` should be confirmed by all authors before journal submission. Names and affiliations should also be checked by all authors.

## Provenance and reuse

Weather source: NASA POWER, https://power.larc.nasa.gov/. Acknowledge NASA POWER and preserve its source metadata when reusing the archive. pvlib, XGBoost and other dependencies retain their respective licenses. The authors' code is released under the MIT License (see `LICENSE` in the repository root).
