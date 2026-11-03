[中文](./reproducibility.zh.md) · [English](./reproducibility.en.md) · [README](../README.en.md)

# Reproducibility guide and public scope

## Three levels of reproducibility

| Level | Supported by this public repository | Entry |
|---|---|---|
| Synthetic demonstration | Run locally, vary inputs and inspect deterministic outputs | `demo/server.py` |
| Method inspection and aggregate cross-checks | Read implementations, captions and released result tables | `research_code/`, `results/`, `figures/` |
| Full empirical rerun | Not possible from this repository alone | Requires authorized data, caches, models and the complete private environment |

The public demo and empirical production chain are separate. Its 12 fictional turbines and analytic paths do not reproduce the 17-turbine, 880-origin forecasting study or the 44-origin execution study.

## 1. Get and run the demo

```sh
git clone https://github.com/zhanghanhnu-coder/wind-window-research.git
cd wind-window-research
python --version
python demo/server.py
```

Use Python 3.10+ and a WebGL browser. The server uses only the Python standard library; research dependencies are unnecessary. Open `http://127.0.0.1:8788`; stop with Ctrl+C. On Windows, double-click `start_demo.cmd` in the repository root.

Alternatively download and extract the [v0.1.0 runtime](https://github.com/zhanghanhnu-coder/wind-window-research/releases/tag/v0.1.0). It is a source runtime requiring Python, not a standalone Windows executable. Documentation in that historical asset may predate main; current instructions live in this repository.

## 2. Run public checks

From the repository root:

```sh
python -m unittest discover -s tests -v
python -m compileall -q demo research_code
```

Six tests cover observed-exceedance priority, unknown versus safe, complete success, equal marginals with different joint probabilities, resource capacity and deterministic synthetic output. `compileall` checks syntax only: it neither imports all dependencies nor runs private experiments. [GitHub Actions](../.github/workflows/checks.yml) runs these checks with Python 3.11.

## 3. Inspect results and figures

- Five main contrasts: [result CSV](../results/aggregate_comparisons.csv). Read direction, metric and bounds together.
- Twenty-six figure sets: [English atlas](experiments.en.md); vectors are in `figures/en/` and `figures/zh/`.
- Titles/captions: [catalog.json](../figures/catalog.json), without original plotting data or evidence paths.
- Open SVGs directly in a browser for vector zooming.

## 4. Research source is not an install-and-run package

`research_code/` archives anonymized research scripts and algorithm implementations for inspection. Some scripts still require packages from the private project (such as `v3`), caches and frozen records. The complete private runtime is not bundled. `research_code/requirements.txt` records research dependencies; installing it does not make every experiment reproducible.

Authorized researchers must restore modules and private input contracts, validate time, turbine order, precision, fingerprints and frozen status, and then choose the applicable scripts. Do not batch-run all `research_*.py`: historical branches may write caches or use older labels. Historical outputs cannot be merged with corrected main results without an explicit distinction.

## 5. Rescoring order in an authorized environment

1. Validate source records, field semantics, quality flags and interval definitions outside this repository.
2. Preserve original float64 truth, missing masks and source fingerprints; confirm node order and chronology.
3. Restore frozen origins, training/calibration boundaries, forecasts and action ledgers.
4. Rebuild three-valued labels, checking exceedance priority and physical targets affected by revisions.
5. Keep forecasts/actions fixed; recompute matched scores and shared-state cost/service bounds.
6. Report before/after ranges, denominators, seasonal composition and failures. Label repair is not model improvement.
7. If changing the model or business rules, create a new version and confirm it on an independent year.

This is an acceptance workflow, not a runnable public command chain. See the [data roadmap](roadmap.en.md); exact node/date requests remain private.

## 6. Minimum checks after a change

For documentation, check relative links, bilingual consistency and comparison directions. For demo changes, run the six tests and inspect the actual page. Algorithm changes additionally require relevant validators in an authorized environment. Passing public CI does not validate empirical conclusions or operational safety.
