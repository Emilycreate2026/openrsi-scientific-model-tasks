# Microkinetic model learning under thermodynamic constraints

Can changes to mechanism selection, constrained parameter learning and restart allocation reduce extrapolation NRMSE beyond the frozen fitted microkinetic model, without weakening physical or structure checks?

12 released transient experiments, 18 possible reversible reaction steps; 8 evaluation conditions, each with 165 time points and four observed species. Evaluation conditions span temperature/feed/program extrapolation.

## Research process

State a mechanism-learning hypothesis; edit the learning procedure; fit using only released measurements; submit the fitted mechanism; use aggregate error and validity status to revise the hypothesis; compare a control or ablation. Repeat across successive code revisions. One numerical optimizer iteration is not an outer research loop.

## Editable boundary

Mechanism-selection/model-learning code, parameterization, loss weighting, regularization, restart allocation and initialization. The public forward equations, measurements, protected checks and evaluation workload are fixed.

## Output and evaluation

mechanism.json, together with candidate training/search source, configuration, and the research log. Judge directly simulates the fitted model; it does not retrain it.

Maximize negative mean NRMSE. For condition c and species s, scale(c,s)=max(range of reference trace, 0.05 * maximum substrate-A feed pressure). e(c)=sqrt(mean over all time/species of ((prediction-reference)/scale)^2). NRMSE=mean over the eight e(c). Score=-NRMSE; dimensionless. It is NOT negative log likelihood.

Finite physical parameters; correct schema and reaction IDs; forward/reverse barrier checks (0.5 kJ/mol numerical allowance); connectivity to both products; mechanism Jaccard >= 0.60; each condition NRMSE <= 0.15 and mean <= 0.08. Parameter limits are log10(A) [-10,25], Ea [0,400] kJ/mol, surface H [-1000,1000] kJ/mol and S [-500,1000] J/mol/K. No gate at the optimized reference value.

## Starting recipe

Create an empty candidate/ directory, then, from the prepared Work root:

```bash
mkdir -p candidate
MKM_WORKERS=4 PYTHONPATH="$PWD/mkm/.." python candidate_source/infer_mechanism.py --data-dir data --out candidate/mechanism.json --report candidate/fit_report.json --workers 4 --max-swaps 0 --max-additions 0
```

Deterministic mechanism screening and constrained numerical refinement in infer_mechanism.py and mkfit.py, with --max-swaps 0 --max-additions 0. The fitted mechanism.json is the evaluation-ready baseline.

## Compute

One physical node. Work: 4 CPU cores, 16 GiB RAM, 0 GPUs. Judge: 4 CPU cores, 8 GiB RAM, 0 GPUs. End-to-end peak: 4 cores and 16 GiB. No GPU type required.

Estimated Work 150 minutes, Judge 2 minutes, plus 5 minutes packaging/snapshot overhead: 157 minutes per cycle; 48 hours / 157 minutes = 18.3 optimistic cycles before extra research time. The archived full reference production completed about 136 minutes after its run-directory timestamp; this is an estimate, not isolated runtime profiling. Work hard cap 10800 seconds; Judge cap 1200 seconds, also 90 seconds per condition.

Read PROTOCOL.md in the author repository for the complete operator-side protocol. Only inputs, candidate_source, reference_artifact and this contract enter Work; the catalyst also receives mkm. No web search, network, external services or external datasets. Local resampling/feature transformations of released training data are allowed; catalyst may simulate its own proposed models with its released forward simulator. Do not read or reconstruct author-only judge assets. Keep source, seeds/configuration, artifact hashes and observations for every outer iteration.
