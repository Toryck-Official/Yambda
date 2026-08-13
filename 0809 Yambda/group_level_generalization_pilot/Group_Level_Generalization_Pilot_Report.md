# Group-Level Generalization Pilot

## Contract

- train targets: 30,258
- validation targets: 4,321
- selected users: 303
- user selection uses train-only stable hashing; no test data was read
- validation history is rolling and contains only groups strictly before each target

## Baseline audit

Global feedback prior means the category distribution of the complete D_SID train split. It is not the Tiny or moderate-Pilot target distribution.
Probabilities: [0.6952056817584402, 0.09252663482418691, 0.1969110821747482, 0.015356601242624677]

## Mark

- full-history train CE: 0.277585
- full-history validation CE: 0.755572
- full-history validation Macro-F1: 0.405722
- last-group validation CE: 0.682509
- last-group validation Macro-F1: 0.411645
- previous-composition validation CE / Macro-F1: 0.656445 / 0.488742
- global full-train prior validation CE / Macro-F1: 0.759805 / 0.212345
- validation CE gain vs strongest simple baseline: -15.10%
- validation Macro-F1 absolute gain: -0.083020

The fixed train-selected full-history checkpoint is worse than both the last-group MLP and previous-group composition baseline. Mean validation Recall for like/dislike/unlike/undislike is 0.9276 / 0.2754 / 0.3117 / 0.0063. It is not formally single-class collapsed, but predictions are severely skewed toward like and nearly never recover undislike.

An explicitly non-decisional curve diagnostic shows that full-history validation CE was best at epochs 2 / 4 / 5 for seeds 2026 / 2027 / 2028 (0.5642 / 0.5611 / 0.5632), while final train-selected checkpoints reached 0.7870 / 0.7283 / 0.7514. Thus early history signal exists, but the frozen 20-epoch procedure strongly overfits. These best-validation points were not substituted as checkpoints after seeing validation.

## Time

- full-history train NLL: 2.118884
- full-history validation NLL: 2.154188
- full-history validation MAE: 37.641619 hours
- full-history validation Median AE: 2.048029 hours
- validation NLL gain vs strongest parametric baseline: 5.50%
- validation MAE gain vs strongest point baseline: 2.65%

The time model passes NLL and MAE but not all metrics: validation Median AE is 2.0480 hours versus 0.7875 hours for the complete-train global median-gap baseline. Time curves also reach their best validation NLL at epochs 7 / 5 / 4 before degrading, so the generalizable time signal is real but accompanied by overfitting.

## Previous-group-size diagnostic

Validation counts by previous-group size are 3,658 / 602 / 58 / 3 / 0 / 0 for bins 1 / 2-5 / 6-20 / 21-100 / 101-500 / >500. All observed slices are finite. There are no validation targets after a >500 group in the moderate Pilot—or in the complete pre-materialized Pilot validation user pool—so >500 stability cannot be evaluated in this round. No burst samples were deleted, and users were not reselected using validation information.

## Decision

- group_level_mark_generalizes: `false`
- group_level_time_generalizes: `true`
- full_history_generalization_supported: `true`
- group_snmpp_design_gate_approved: `true`

Approval is narrow: the full-history time signal generalizes, so a future design gate may be investigated. Mark-history generalization is not supported by the frozen procedure, and >500 validation stability remains unverified. This is not approval to implement or train Group-SNMPP.

This pilot stops here. It does not design or train Group-SNMPP.
