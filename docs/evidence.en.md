[中文](./evidence.zh.md) · [English](./evidence.en.md) · [README](../README.en.md)

# Main results and limits of inference

This page lists public aggregate values and how to read them. Detailed captions are in the [26-set atlas](experiments.en.md). Values come from frozen research, not the synthetic demo. Five main comparisons are also in [aggregate_comparisons.csv](../results/aggregate_comparisons.csv).

## Results

| Experiment / direction | Metric and range | Supported interpretation | Unsupported interpretation |
|---|---|---|---|
| Graph − same-24h mean | Brier: [-0.006691, -0.004739] | Graph helps in the matched-information contrast | Graph beats every strong baseline |
| Original time structure − time shuffle | Brier: [-0.069800, -0.036301] | Temporal coupling helps overall | It helps every season |
| Same contrast, September–October | Brier: [+0.023911, +0.028050] | A directional reversal exists | The reversal may be dropped to claim general superiority |
| Matched-window update interaction | [+0.046410, +0.084210] | Positive pooled interaction | Stable across seasons; excluding September–October crosses zero |
| Rolling28 − previous-month fixed | Brier: [-0.069124, -0.045564] | Lower pooled window loss | Every decision metric improves |
| Same contrast, false-accept share | [-0.104278, -0.085344] | Fewer erroneous acceptances | The same change in a conditional false-positive rate |
| Same contrast, missed-window share | [+0.019485, +0.038419] | More feasible opportunities are lost | Rolling calibration has no tradeoff |
| Analog − graph, 72h, fixed resources | Model cost: [-47.536301, +63.069760] | Overall cost direction is unidentified | Proven equality or stable savings |
| Same main comparison | Successful tasks: [-0.416667, +0.401515] | Service advantage is unidentified | Cost bounds substitute for service evaluation |

False-accept and missed-window shares both use all 59,840 single-turbine forecast records as denominator. They are not conditional rates among unsafe or feasible windows. Cost/service evaluation uses all 44 execution origins, not a ranking restricted to 15 complete-weather cases.

## Reading a range

1. Read the direction: for “analog − graph”, negative cost favors the analog.
2. Read the metric: lower Brier is better; more successful tasks is better.
3. Check zero crossing. Crossing zero means known facts admit both directions; it does not establish equivalence.
4. Check the range type. These are missing-truth identification ranges, not confidence or credible intervals.
5. Check pooled and seasonal results together instead of selecting favorable subgroups.

## Tighter evidence is not new observation

The coarse main cost width is 186.9697. Sharing physical hourly states reduces it to 110.6061. If the first 48 necessary states are validly resolved, worst-case remaining width is at most 71.1364, guaranteeing at least 35.68% further reduction relative to 110.6061. This is not a 35.68% cost saving and does not guarantee a nonzero ranking.

## Engineering denominators

The fixed matrix has four inputs (two instances × two weight representations), two algorithms and two formulations: 16 requests. The original formulation completed 5/8 and the bounded formulation 8/8. Five mutually completed pairs have zero objective difference; three failed references have no paired value. This is neither a universal 100% success rate nor eight independent validated field cases.

## Counterexamples to retain in the manuscript

- The 112-day analog beats the graph: distinguish structural gain from information budget.
- Temporal dependence reverses in September–October: pooled evidence is heterogeneous.
- Absolute scores degrade under synchronized input outages: relative advantage is not recovery.
- Main fixed-resource cost and service bounds cross zero: forecast gains do not establish savings.
- Solver failures stay in the denominator: reporting only completed cases is selective.

Per-origin outputs and identifiable acquisition requests are withheld. This page supports interpretation and methodological inspection, not independent data-free recalculation of empirical values.
