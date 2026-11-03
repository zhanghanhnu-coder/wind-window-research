[中文](./contributing.zh.md) · [English](./contributing.en.md) · [README](../README.en.md)

# Contributing and maintaining public materials

Contributions to bilingual documentation, legends, accessibility, the synthetic demo and algorithm review are welcome. New experiments must distinguish synthetic validation from observed-data results.

## Before submitting

- State contrast direction, evaluation population, metric and range type; update both languages.
- Explain the triggering condition, intended behavior and validation for code changes.
- Run [public checks](reproducibility.en.md) and inspect the page after demo changes.
- Identify the branch/input contract for research changes; synthetic tests alone do not establish empirical reproduction.
- Use explicit units, legends and denominators. Do not label identification bounds as confidence intervals.

## Do not submit

Raw or cleaned time series, field dictionaries, real site names/coordinates, weights, private caches, per-origin outputs, action ledgers, acquisition details, personal records, credentials or screenshots containing them. Renaming a site does not authorize data publication. Use small fictional examples when reporting problems.

## Reporting an issue

Include the revision, Python/browser versions, command, expected/actual behavior and sanitized error information. Do not paste entire research logs or local paths. Data requests belong in authorized private channels, not public attachments.

Project code and documentation use the [MIT license](../LICENSE); third-party components retain their own terms. Data are outside the license scope. Academic references should record the repository URL, commit and access date; no unverified paper DOI or publication status is supplied.
