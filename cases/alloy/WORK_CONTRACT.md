# Sparse multitask surrogate learning for alloy extrapolation

Can alternative shared representations and censoring-aware objectives improve the four-property predictor beyond the frozen physics-structured model, particularly for unseen Ti-bearing compositions and longer aging times?

192 partially labeled training specimens and 300 candidate specimens: 100 interpolation, 80 element holdout, 120 process holdout. Four target properties, with a fixed design-window membership target. No active-label acquisition or 2,000-call simulator budget.

## Research process

The outer AutoResearch loop edits the surrogate learning procedure, fits on the released data, submits predictions, receives aggregate score and validity, then changes the next hypothesis based on feedback. Require a baseline/control comparison and at least one ablation of sharing or censoring. This is model-development experimentation, not a sequential acquisition loop within a fixed policy.

## Editable boundary

Descriptors, cross-property sharing, model family, censoring/missing-label losses, regularization, fitting and ensembling. Training rows, candidate rows, reference property values, null constants, group assignment and metric are fixed.

## Output and evaluation

predictions.csv plus the reproducible learning source/configuration and research log. Judge directly evaluates the prediction artifact; no Judge-side training and no candidate-provided score is trusted.

Maximize mean over the four properties of skill_p = 1 - MAE_p/MAE_null,p, evaluated over all 300 candidates. Constant null values: yield_strength_mpa 742.5, elongation_pct 19.0, hardness_hv 329.5, sigma_fraction 0.0. The mean uses each property once; subgroup metrics are not duplicated in the aggregate. Score is dimensionless.

Exact candidate identifiers, finite bounded predictions and binary meets_spec. Feasible ranges: yield [0,5000], elongation [0,150], hardness [0,2000], sigma [0,1]. Skill minima by interpolation/element/process: yield .50/.50/.45; elongation .50/.50/.45; hardness .45/.40/.30; sigma .40/.40/.40. Overall design-window F1 >= .40. Window: yield >=700 MPa, elongation >=10%, sigma <=.02. No gate at the optimized reference value.

## Starting recipe

Create an empty candidate/ directory, then, from the prepared Work root:

```bash
mkdir -p candidate
ALLOY_DATA_DIR="$PWD/data" ALLOY_RESULTS_DIR="$PWD/candidate" python candidate_source/solve.py
```

Physics-structured multitask fitting in baseline/source/solve.py: composition descriptors, phase modeling, shared reduced-time kinetics, property coupling and censoring-aware elongation loss. The frozen predictions.csv is evaluation-ready. No implemented expected-improvement acquisition baseline is claimed.

## Compute

One physical node. Work: 4 CPU cores, 8 GiB RAM, 0 GPUs. Judge: 2 CPU cores, 4 GiB RAM, 0 GPUs. End-to-end peak: 4 cores and 8 GiB.

Estimated Work 15 minutes, Judge 1 minute, plus 5 minutes overhead: 21 minutes per cycle; 48 hours / 21 minutes = 137.1 optimistic cycles before extra research time. The archived baseline is a small deterministic numerical fit; this allocation is conservative, not a claimed measured training runtime. Work cap 1800 seconds; Judge cap 1200 seconds.

Read PROTOCOL.md in the author repository for the complete operator-side protocol. Only inputs, candidate_source, reference_artifact and this contract enter Work; the catalyst also receives mkm. No web search, network, external services or external datasets. Local resampling/feature transformations of released training data are allowed; catalyst may simulate its own proposed models with its released forward simulator. Do not read or reconstruct author-only judge assets. Keep source, seeds/configuration, artifact hashes and observations for every outer iteration.
