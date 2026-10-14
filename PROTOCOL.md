# OpenRSI evaluation and research contract

These are three separate research tasks, each selected independently. This is
an auditable source overlay with existing datasets, fitted reference artifacts,
source and executable evaluation, not a claim of OpenRSI acceptance.

## Evaluation mode and score

Use direct candidate-only evaluation. Solution runs `tools.py materialize`:
it only copies an existing fitted artifact, without training or evaluation.
Candidate-producing fitting and research take place in Work. The case-specific
WORK_CONTRACT.md defines the metric, all original scientific checks and workload.
The historical exact-best binary gates are not part of this OpenRSI adaptation.
Valid candidates receive a continuous score; higher is better in all three
tasks (the catalyst score is negative NRMSE). Passing validity alone is not a
scientific improvement. Compare the score to the frozen reproduced baseline.
The treatment's logged-care comparator is part of the fixed simulator workload,
not a retrained or paired submitted baseline.

Malformed/missing artifacts, scientific-constraint failures, attributable
compile failures, crashes and illegal memory accesses are unscored
(`candidate_invalid`, score null). Timeout, incomplete evaluation and
infrastructure failure are separately unscored, never silently assigned zero or
retained as successes. Evaluate the complete workload exactly once for each
candidate; no best-of-seed selection, test-case replacement or dropped cases.
There is no requirement for GPU training or multiple training seeds. Any extra
candidate training seed, retry, resampling or failed candidate consumes the
Work/total wall-clock budget. Infrastructure reruns must be recorded separately,
preserve the same candidate hash and never be treated as adaptive improvement.

## Snapshots and authority

The operator prepares Base with `tools.py prepare-work`. Never put this entire
author checkout, its git history, judge files, generator, truth tables or
author-only metric logs in Base/Work. Work contains released data, learner source
and the reference artifact. Judge reloads the COMPLETE materialized Work snapshot
using current RSI-Harness semantics. Task-owned tests and reference assets are
injected Judge-only, root-owned and not writable/readable by candidate code.
This is not an independent clean-Base evaluation mode. Verify protected runtime,
input, simulator and test hashes before running Judge; reject modified protected
components. Do not trust candidate-written metrics or stdout as the score.
Use the evaluator's own parsed output. The author CLI is for trusted reference
reproduction; it is not a standalone sandbox for arbitrary untrusted code.

The executable treatment policy runs as an unprivileged uid, with a sanitized
environment, process/memory/file-size limits, process-group timeout and strict
actions-schema checks. In the actual Harness lane, deny networking, capabilities,
privilege escalation, host mounts and access to evaluator-owned files. The
catalyst and alloy submissions are parsed data, not executable evaluator code.
Permissions and hashes reduce tampering; they do not prove arbitrary candidate
code safe under every shared-snapshot implementation. Validate that boundary
when building the actual Harbor task, using the existing Harness mechanisms.

## Feedback and exposure

Return only feedback.json (score/status/timing) and bounded schema/validity
diagnostics without true parameters, labels, patient outcomes or per-example
residuals. Retain detailed author-only logs for maintainers, not for the agent.
Do not inject author reproduction records or per-case reference metrics into
Work. Public baseline artifact availability is intentional: the research goal
is to improve that reference, not recover a secret baseline.

The author audit repository is publicly inspectable, INCLUDING evaluation assets.
Thus "Judge-only" means unavailable during the offline task, not secret from the
world. Offline isolation cannot rule out prior exposure, memorization, or adaptive
overfitting to repeated aggregate feedback. Do not claim this is a sealed test or
proof of generalization to new data. Judge uses this fixed workload; no unprovided
future private dataset or service is required. Preserve complete trajectories and
review source/ablations for hard-coded answers or evaluation-content copying.

## Data and network

The three datasets are synthetic scientific-model fixtures, not patient records
or real alloy/catalyst measurements. No accounts, API keys or external services
are needed. Web search is disabled. All Work and Judge network access is disabled
after image build. Dependencies are installed during operator setup only.
Additional external data and additional pretrained models are disallowed.
Allow only training-data resampling/augmentation/features and, for the catalyst,
simulations of candidate-owned models using the released forward equations.
The treatment and alloy truth-generating models are not research APIs.

## Research trajectory and uncertainty

Default trajectory budget is 24 hours, extendable to 48 hours. Each case states
phase-specific caps and estimates. Judge time pauses Work and still counts in
the trajectory. Include 5 minutes per-cycle packaging/snapshot overhead in the
planning estimate. These estimates are not a measured guarantee of loop counts.
Record hypothesis, source revision, seeds, command, artifact hashes, raw evaluator
feedback, time and decision for every candidate, including failures and retries.
Use controls/ablations under identical data and compute boundaries. Direct artifact
evaluation is deterministic on the pinned workload; the historical paired replays
and the new artifact rechecks support reproducibility. No claim of universal
scientific significance follows from a tiny gain on these synthetic cases.
Repeat training only when its variability could change the comparison; assess
stability and practical effect size rather than impose an invented universal
minimum improvement. Terminate NaN, OOM, crash and time-limit violations clearly.
