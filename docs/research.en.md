# Research methods and progress

The proposal follows graph construction → 72-hour probabilistic forecasting → robust resource scheduling → integrated simulation. The implemented auditable route uses temporal graph aggregation and joint scenarios. Statistical dependence is not verified physical propagation, and not every originally proposed architecture has been implemented.

Three-valued truth: any observed exceedance establishes failure; success requires all hours observed and below threshold; otherwise unknown. Thresholds 8/10/12 use unverified original units. Training, calibration and evaluation are temporally separated; future truth is reserved for retrospective evaluation.

Forecasting uses 880 origins and 17 turbines, with 392 complete-input origins. Execution uses 44 origins; the seasonally selected 15 complete-weather origins are auxiliary. 59,840 single-turbine prediction records represent 16,269 physical targets. Repeated windows and seeds are not additional independent weather samples.

H1 contrasts graph, mean and own-history models under the same 24-hour information, with 28-day climate and 112-day analog strong baselines. H2 permutes temporal/spatial coupling while preserving marginal multisets, assessing single/joint six-hour events. H3 crosses fixed/rolling calibration, one-step/72-hour planning and fixed/own booking. Both calibration arms receive current forecasts every six hours. Residual28 is a separate branch and was not used in that execution cross.

Shared turbine-hour constraints tighten cost identification ranges. Cost and successful-task bounds are optimized separately, not a joint rectangle. Penalty sensitivity revalues fixed actions instead of reoptimizing them. Scenario grouping/pruning and bounded formulations are verified only on saved engineering instances.

Completed modules: label/data governance, predictors and baselines, joint/missingness experiments, execution/information value, prototype and manuscript figures. Next: verify field/time semantics → rescore unchanged predictions/actions → freeze protocol → validate on independent years and genuine operational/metocean data. Further same-year model searching is paused.

Proposal term: June 2026–May 2028; original midterm milestone May 2027. Materials prepared November 3, 2026; formal review date remains unconfirmed. Publication and intellectual-property outputs remain future goals.

[Experiments and captions](experiments.en.md)
