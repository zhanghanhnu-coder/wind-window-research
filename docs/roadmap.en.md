[中文](./roadmap.zh.md) · [English](./roadmap.en.md) · [README](../README.en.md)

# Next steps and required data

## Order of work

| Stage | Required input | Work | Completion evidence |
|---|---|---|---|
| Semantic acceptance | Field definitions, quality codes and time conventions | Confirm variable, units, scaling, sensor context and intervals | Auditable data contract |
| Frozen rescoring | Valid first-48 and complete 234/146 requests | Keep forecasts/actions; revise truth and ranges | Before/after labels, width changes and failures |
| External confirmation | Independent year, genuine metocean and business rules | Apply a frozen forecasting/execution protocol | Untouched evaluation and applicability report |
| Paper and research outputs | Complete evidence chain | Manuscript, software and IP applications | Verifiable submission/publication/registration records |

Until new evidence arrives, further model selection on 2024 data is paused. Documentation and synthetic software can improve, but are not new empirical validation.

## Concrete data needs, public version

| Priority | Material | Acceptance requirements | Purpose |
|---|---|---|---|
| P0 | Five-minute records for the first 48 and full 234 necessary turbine-hours | Twelve original readings per hour, original timestamps, quality codes, source files and revisions; not just hourly means | Main cost-comparison truth |
| P0 | `feature_A` semantics | Name, units, scaling, sensor position/height, aggregation, abnormal codes, interval start/end convention, timezone and daylight-saving rules | Threshold interpretation and alignment |
| P1 | Records for the other 146 retained requests | Same original-value and quality requirements; retain even if unimportant to the main contrast | Other fair execution contrasts |
| P1 | Metocean/weather | Synchronized significant wave height, period, wind direction and acquisition/availability times | Operational applicability |
| P1 | Vessels, crews, tasks and costs | Vessel limits, crew rules, transit time, actual task start/end/interruption, failure causes, resource/cost ledgers | Executor and cost-model validation |
| P2 | A full year unused in development | Consistent raw records for 17 turbines, quality/missingness logs and authoritative node mapping | External validation |
| P2 | Historical forecasts and authoritative layout | Forecast versions actually available at each issue time, publication times and authorized location/distance information | Leakage-free replay and propagation assessment |

The 48/234/146 counts describe existing private request lists, not publicly identifiable rows. Exact nodes, dates, original field codes and request files must use authorized private channels, not public Issues or PRs.

## What to do when records arrive

Validate incoming originals before examining revised rankings. Rescore frozen actions, reporting cost and service separately. If verified units change the scientific definition, create an explicit new definition/version while retaining the historical proxy-scale study and its limits; do not silently overwrite conclusions.

If only some records arrive, retain unresolved states and recompute conditional ranges. If an independent year remains unavailable, frame the paper as a development-data method study, not external confirmation.

## Planned timing and status

Late 2026 prioritizes documentation and semantics; early 2027 frozen rescoring; late 2027 independent-year and real-operation evaluation where feasible; 2028 manuscript/software outputs and completion. These dates are conditional on data availability. No unverified award level, project number, funding or formal review date is asserted here.
