# Policy program interface

The submission is the directory `/app/submission`. It must contain an
executable Python program `policy.py` and a file `value_estimate.json`. Any
other files the program needs (fitted parameters, helper modules) must live in
the same directory and be located relative to `policy.py`.

## Invocation

    python3 /app/submission/policy.py --patients <patients.csv> --history <history.csv> --output <actions.csv>

* `<patients.csv>` has the schema of `/app/data/patients.csv` for the cohort
  currently under treatment (new patients, new identifiers).
* `<history.csv>` has the schema of `/app/data/trajectories.csv` and contains,
  for every patient in `<patients.csv>`, one row per decision point from `step`
  0 up to and including the current decision point t. Rows for the current
  decision point have empty `action` and `step_cost` fields; all earlier rows
  are complete. Every patient in the cohort is at the same decision point.
* `<actions.csv>` must be written with header `patient_id,action` and exactly
  one row per patient in `<patients.csv>`, `action` in {0, 1, 2}.

## Execution contract

The program is invoked six times per cohort (once per decision point), on
cohorts of up to 6000 patients, in a container with the same Python interpreter
and package versions as this environment, without network access, as an
unprivileged user whose only writable locations are the output path and a copy
of the submission directory, with `OMP_NUM_THREADS=1` (single-threaded numeric
libraries), an address-space limit of 4 GiB and a wall-clock limit of 300
seconds per invocation. The verifier compares the actions from repeated
invocations on identical inputs; they must be identical. The submission
directory may hold at most 200 files and 64 MiB in total and may not contain
symbolic links.

## value_estimate.json

    {"estimated_utility": <number>}

Your estimate of the expected `utility` that `policy.py` achieves on a new
patient drawn from the same population as `/app/data`.
