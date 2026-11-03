# xx风电场: probabilistic windows and robust wind O&M

[中文](README.md) | [Methods and progress](docs/research.en.md) | [Digital twin](docs/twin.en.md) | [Figure catalog](figures/catalog.json)

This project studies the chain from weather inputs and joint continuous-window probabilities to resource-constrained maintenance execution. The confidential study site is named **xx风电场** throughout this release.

## Status — 2026-11-03

Completed work includes float64 label reassessment, fair and strong baselines, dependence interventions, calibration, missing-input stress, matched forecast updates, prediction–execution crosses, cost/service identification, targeted information acquisition and frozen-instance solver checks. A research manuscript draft and digital-twin prototype exist; publication, granted patents or software registration are not established by this repository. All 2024 observations remain development material. Field semantics, metocean limits, operational records and independent-year validation remain unavailable.

| Comparison | Aggregate range | Interpretation |
|---|---|---|
| Graph − same-24h mean model | Brier [-0.006691,-0.004739] | Better than fair mean baseline, worse than strong 112-day analog |
| Original − time-shuffled dependence | [-0.069800,-0.036301] | Pooled improvement; reversal in September–October |
| Matched-update interaction | [0.046410,0.084210] | Positive overall; crosses zero when September–October is removed |
| Analog − Graph cost | [-47.536301,63.069760] | Overall ranking unidentified |
| Analog − Graph successful tasks | [-0.416667,0.401515] | Service advantage unidentified |

These are missing-truth identification ranges, not confidence intervals. Costs are model units. Resolving all first 48 requested states out of 234 essential states guarantees at least 35.68% narrowing of the primary range; this is conditional information value, not realized savings or recovered observations.

## Run

Python 3.10+, no third-party Python dependency:
```sh
python demo/server.py
```
Open http://127.0.0.1:8788 or double-click `start_demo.cmd` on Windows. Three-dimensional assets are bundled. Locations, weather and forecast scenarios are entirely synthetic. Run `python -m unittest discover -s tests` for label and window checks.

## Reproducibility and release boundaries

`research_code/` archives sanitized original research code. Most scripts require withheld private caches and cannot reproduce the empirical results without authorized data. `demo/` is a separate synthetic public demonstration, not the original fitted predictor or optimizer. `results/` contains aggregate comparisons only. `figures/` contains 26 bilingual figure sets: two of the original 28 sets locating real turbine/date acquisition requests are excluded.

No raw or cleaned measurement series, point dictionaries, real coordinates, fitted weights, per-origin predictions, task ledgers, acquisition lists, private caches, applications or review forms are included. Original figure metadata, evidence folders and previous ZIPs are not copied. Renaming a site does not make raw observations publishable. Keep private data outside this repository.

Project code and documentation: MIT. Bundled Three.js retains its MIT notice. Dependency licenses apply separately; private data are not licensed here.


Private field codes are replaced by feature_A/B/C/D. Authorized users must supply their own field mapping and input caches.
