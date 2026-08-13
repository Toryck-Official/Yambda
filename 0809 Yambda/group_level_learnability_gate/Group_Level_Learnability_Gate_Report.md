# Group-Level Learnability Gate Report

## Protocol

The gate reuses exactly the frozen 6,413 train target groups. Evaluation is on the same Tiny train subset and therefore establishes optimization/learnability only, not validation or test generalization. No within-group order, listen, SNMPP, SID target, HPN, or BOLA is used.

## Data

- target groups: 6,413
- target events: 38,363
- users: 6,295
- strict-history group union: 1,159,633
- representation: mean frozen SID semantics + feedback composition + log group size + previous log gap

## Mark-only result

- recurrent CE/event: 0.489052 +/- 0.000955
- recurrent Macro-F1: 0.766283 +/- 0.003119
- recurrent Accuracy: 0.808592 +/- 0.001405
- mean per-class Recall (like/dislike/unlike/undislike): 0.852454 / 0.699525 / 0.881699 / 0.617917
- last-group MLP CE/event: 0.671588
- last-group MLP Macro-F1: 0.686901
- global prior CE/event: 1.711381
- previous composition CE/event: 1.389574
- shared-group oracle floor: 0.245290 CE/event
- CE gain over the strongest simple baseline (last-group MLP): 27.18%
- collapsed recurrent seeds: 0 / 3

The recurrent model predicts all four classes in all three seeds. Thus the earlier one-class collapse is not an unavoidable property of the grouped target.

## Time-only result

- recurrent NLL/group: 2.171074 +/- 0.025061
- recurrent MAE: 39.489229 hours
- recurrent Median AE: 2.211899 hours
- history-free lognormal NLL: 2.549346
- previous-gap residual lognormal NLL: 2.604977
- global train median-gap MAE / Median AE: 41.871205 / 0.788889 hours
- previous-gap MAE / Median AE: 62.370924 / 9.536111 hours
- NLL gain over the strongest parametric baseline: 14.84%
- MAE gain over the strongest point baseline: 5.69%

Time is therefore learnable under NLL and mean absolute error, but the recurrent model does **not** beat the global median baseline on Median AE. It reduces some long-tail errors while making the typical absolute error larger. This is mixed evidence, not uniformly superior time prediction.

## Joint sanity

Because both single tasks passed the predeclared gate, a simple shared-GRU model with independent mark/time heads was run. Across three seeds it remained finite and non-collapsed:

- mark CE/event: 0.549234
- mark Macro-F1: 0.741527
- time NLL/group: 2.204509
- time MAE: 39.774989 hours

Joint optimization is stable but weaker than the corresponding independently trained heads, so it does not replace the single-task evidence.

## Previous-group-size stability

All size slices produced finite outputs. For the 17 targets following a `>500` group:

- mark CE/event: 0.061438
- mark Macro-F1: 0.507818
- time NLL/group: 0.929497
- time MAE: 19.672045 hours

There is no numerical explosion from `>500` groups under mean-pooled group representation. The slice is too small and contains no undislike target events, so its class metrics are diagnostic only and cannot establish better predictive quality for extreme bursts.

## Decision

- mark learnable: `true`
- time learnable: `true`
- history signal supported: `true`
- group SNMPP design approved: `true`

Here, approval means only that a future group-aware method is scientifically testable: the grouped task and full-history signal are learnable on the same Tiny train subset. It does not approve a specific Group-SNMPP architecture and does not establish validation/test generalization.

The gate stops here. It does not design or train Group-SNMPP.
