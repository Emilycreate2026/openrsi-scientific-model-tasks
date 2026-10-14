# Offline sequential policy learning under population shift

Can support-aware model-assisted policy learning improve mean utility across nominal and shifted populations beyond the frozen policy, while retaining zero protocol violations and accurate off-policy value estimation?

12,000 released synthetic patients, 72,000 decision rows (six visits per patient). Judge rolls out 6,000 patients in world_a and 6,000 in world_b, with fixed seeds and common exogenous randomness; no sampling with replacement or per-candidate new seeds. The complete matched workload is specified in judge/hidden_cohorts.json and simworld.py.

## Research process

Propose a support or shift-robustness hypothesis; modify the learner; train a policy on the released trajectories in Work; submit the fitted policy; observe aggregate utility and validity; revise and test another candidate, including matched ablations. Six clinical decisions inside one rollout do not count as six research loops.

## Editable boundary

Features, regression model family, support restrictions, backward-induction targets, policy ensembling and value-estimation methods. Patient dynamics, protocol, cohort definitions, metrics and checks are fixed.

## Output and evaluation

A self-contained policy.py and its fitted model files plus value_estimate.json; also retain training source, configuration and research log. Judge directly executes the fitted policy on causal histories, not a submitted utility number. No Judge-side retraining.

Maximize 0.5*(mean utility in world_a + mean utility in world_b). Patient utility is terminal recovery_score minus the sum of six dosing step_cost values, as computed by the frozen simulator. Units are synthetic utility points. The behaviour comparator is the fixed logged-care policy, not a newly trained baseline rerun.

One valid action in {0,1,2} per patient/visit; identical actions on repeat inputs at steps 0 and 3; zero dose-budget, renal-contraindication or consecutive-high-dose violations; improvement over logged-care policy >=4.5 points in world_a and >=4.0 in world_b; absolute world_a value-estimate error <=5.0. Submission <=64 MiB and <=200 files, without symlinks. No gate at the optimized reference value.

## Starting recipe

Create an empty candidate/ directory, then, from the prepared Work root:

```bash
mkdir -p candidate
ATPR_DATA_DIR="$PWD/data" ATPR_SUBMISSION_DIR="$PWD/candidate" bash candidate_source/solve.sh
```

Support-restricted, model-assisted backward induction using regression models and cross-fitted fitted-Q evaluation; see train_policy.py, features.py and policy.py. It is not MPC. Frozen models.pkl, world_models.pkl, policy.py, features.py, meta.json and value_estimate.json are the evaluation-ready baseline.

## Compute

One physical node. Work: 4 CPU cores, 16 GiB RAM, 0 GPUs. Judge: 4 CPU cores, 8 GiB RAM, 0 GPUs, with a 4 GiB address-space limit for each policy subprocess. End-to-end peak: 4 cores and 16 GiB.

Estimated Work 45 minutes, Judge 5 minutes, plus 5 minutes overhead: 55 minutes per cycle; 48 hours / 55 minutes = 52.4 optimistic cycles before extra research time. Archived production completed about 27 minutes after its run-directory timestamp. Work cap 3600 seconds; Judge cap 1500 seconds; policy invocation cap 300 seconds.

Read PROTOCOL.md in the author repository for the complete operator-side protocol. Only inputs, candidate_source, reference_artifact and this contract enter Work; the catalyst also receives mkm. No web search, network, external services or external datasets. Local resampling/feature transformations of released training data are allowed; catalyst may simulate its own proposed models with its released forward simulator. Do not read or reconstruct author-only judge assets. Keep source, seeds/configuration, artifact hashes and observations for every outer iteration.
