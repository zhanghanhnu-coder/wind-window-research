[中文](./code-map.zh.md) · [English](./code-map.en.md) · [README](../README.en.md)

# Code map and reading order

## Separate the two entry points

`demo/server.py` starts the runnable public synthetic demonstration. `research_code/` is an anonymized research archive, not the demo entry or a complete installable package. Some scripts require unpublished internal modules, frozen caches and models. Read core functions before orchestration scripts; historical model names in filenames do not define current conclusions.

## Research code map

| Module | Core/entry | Related script | Focus |
|---|---|---|---|
| Truth and windows | [precision_truth_core.py](../research_code/precision_truth_core.py) | [research_precision_revision.py](../research_code/research_precision_revision.py) | Original precision and three-valued events; model tensors separate from truth |
| Matched and strong baselines | [review_baselines_core.py](../research_code/review_baselines_core.py) | [research_review_baselines.py](../research_code/research_review_baselines.py) | Information budgets, input masks and window probabilities |
| Dependence interventions | [closure_joint_core.py](../research_code/closure_joint_core.py) | [closure_joint.py](../research_code/closure_joint.py) | Time/space permutations with invariant marginal multisets |
| Input outages | [closure_outage_core.py](../research_code/closure_outage_core.py) | [closure_outage_run.py](../research_code/closure_outage_run.py) | Matched complete-input cohort with synchronized masking |
| Fixed/rolling calibration | [review_fixed_rolling_pair.py](../research_code/review_fixed_rolling_pair.py) | [verify_review_fixed_rolling_pair.py](../research_code/verify_review_fixed_rolling_pair.py) | Calibration comparison and paired verification |
| Update robustness | [closure_robustness.py](../research_code/closure_robustness.py) | [closure_robustness_independent_review.py](../research_code/closure_robustness_independent_review.py) | Matched targets, physical weighting and season-deletion diagnostics |
| Execution and resources | [causal_execution_core.py](../research_code/causal_execution_core.py) | [review_cross_execution.py](../research_code/review_cross_execution.py) | Irrevocable attempts and fixed-resource crosses |
| Shared-state bounds | [iteration2_bounds_core.py](../research_code/iteration2_bounds_core.py) | [iteration2_analog_execution.py](../research_code/iteration2_analog_execution.py) | One unknown state per physical turbine-hour |
| Acquisition value | [iteration3_acquisition_core.py](../research_code/iteration3_acquisition_core.py) | [iteration3_acquisition_core.py](../research_code/iteration3_acquisition_core.py) | Worst remaining width for a fixed query |
| Bounded formulation | [bounded_distribution_core.py](../research_code/bounded_distribution_core.py) | [research_bounded_distribution.py](../research_code/research_bounded_distribution.py) | Finite variable bounds using nonnegative cost structure |
| Scenario grouping | [scenario_equivalence_core.py](../research_code/scenario_equivalence_core.py) | [research_scenario_equivalence.py](../research_code/research_scenario_equivalence.py) | Grouping scenarios equivalent in constraints/cost |
| Dominated support | [support_dominance_core.py](../research_code/support_dominance_core.py) | [research_support_dominance.py](../research_code/research_support_dominance.py) | Engineering checks for zero-mass dominated support |

## Suggested reading path

1. Read `precision_truth_core.py` for truth precision and chronology assertions.
2. Read `review_baselines_core.py` and `closure_joint_core.py` to separate marginals, scenario ranks and continuous events.
3. Read `causal_execution_core.py` and related planning code to separate issue-time information, irrevocable actions and retrospective evaluation.
4. Read `iteration2_bounds_core.py` for window AND terms and shared physical states.
5. Read `iteration3_acquisition_core.py` for worst remaining width under a fixed query.
6. Read solver-engineering branches last; numerical-formulation improvements are not forecast improvements.

## File categories

| Pattern | Typical role | Caveat |
|---|---|---|
| `*_core.py` | Mathematics, contracts or execution logic | Not all are standalone imports |
| `research_*.py`, `review_*.py`, `closure_*.py` | Stage-specific orchestration and reports | May write caches or use historical conventions; do not batch-run |
| `verify_*.py`, `*_verify.py` | Branch-specific assertions/certificates | Require complete private branch inputs |
| `*_tables.py`, `*_report.py` | Tables and reports | Cannot rebuild absent source tables |
| `tests/test_demo.py` | Public synthetic contracts | Does not validate research models or empirical claims |

Some files retain historical model names, directory structures and experiment settings. The archive records implementation evolution; current interpretation follows [methods](research.en.md) and [evidence](evidence.en.md). Anonymous `feature_A/B/C/D` identifiers are not a real field dictionary; authorized users must map them privately.
