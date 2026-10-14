# Reproducible scientific model-development tasks for OpenRSI

Source and baseline evidence for OpenRSI Index discussions #222 (catalyst),
#223 (treatment), and #224 (alloy). Each case is an independent task.

Contributor: Huaiming Li, algorithm engineer, Famou Agent Team, Baidu AI Cloud.
Contact: 18811607325@163.com. GitHub account: Emilycreate2026.
Experience claimed here is algorithm engineering, executable evaluation and
reference-solver reproduction, not clinical or experimental-science credentials.

## Quick reproduction (Linux, Python 3.12)

```bash
python -m pip install -r requirements.txt
python tools.py verify
python tools.py materialize catalyst --out /tmp/catalyst-reference
python tools.py evaluate catalyst --candidate /tmp/catalyst-reference --out /tmp/catalyst-evaluation
```

Replace `catalyst` with `treatment` or `alloy` and use fresh output paths.
All commands operate on actual checked-in inputs, evaluators and fitted reference
artifacts. `materialize` performs no training or evaluation. Treatment artifacts
include Python pickle model files; these are trusted baseline assets whose hashes
are pinned in ASSET_SHA256.json, not files to load from arbitrary third parties.

To create an agent-visible starting directory:

```bash
python tools.py prepare-work alloy --out /tmp/alloy-work
```

Use the recipe in its README.md to fit a new candidate. Do not give the agent the
whole source checkout. PROTOCOL.md defines complete-snapshot Work/Judge handling,
data visibility, networking, invalid-run behavior and residual public-exposure
limitations. Author-side local rechecks do not certify an RSI-Harness sandbox.

## Frozen reference results

| Case | Raw metric | Direction |
| --- | ---: | --- |
| catalyst | 0.006783494785813174 mean NRMSE | lower |
| treatment | 51.46555166666667 mean utility | higher |
| alloy | 0.871242468340986 mean property skill | higher |

Every reference passes the original scientific checks. Historical double replay
records and a fresh direct-artifact recheck are in each baseline directory. The
new check did not retrain the model or claim a new score improvement. Repository
paths, exact source commit, phase budgets and methods appear in each proposal.
