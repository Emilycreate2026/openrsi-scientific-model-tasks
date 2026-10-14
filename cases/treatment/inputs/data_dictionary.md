# Data dictionary: observational treatment trajectories

All records are synthetic. They were produced by rolling a logged care policy
through a private physiological simulator; the simulator, its parameters and the
patients' latent physiological states are not released. Each patient has one row
in `patients.csv`, six rows in `trajectories.csv` (decision points `step` 0..5,
one every treatment interval) and one row in `outcomes.csv`.

## patients.csv (baseline, one row per patient)

| column | type | meaning |
|---|---|---|
| `patient_id` | int | unique identifier; carries no information |
| `age` | float | years |
| `weight_kg` | float | body weight, kilograms |
| `comorbidity` | int 0..4 | count of chronic comorbid conditions |
| `frailty_score` | int 1..7 | clinical frailty scale assessed at admission (higher = frailer) |
| `culture_flag` | int 0/1 | baseline culture result: 1 = pathogen reported susceptible to the drug. The assay is imperfect in both directions. |

## trajectories.csv (one row per patient and decision point)

| column | type | meaning |
|---|---|---|
| `patient_id` | int | joins to `patients.csv` |
| `step` | int 0..5 | decision point index |
| `inflam_marker` | float | inflammation biomarker, log scale (higher = more inflammation) |
| `renal_marker` | float | renal function marker (higher = worse renal function) |
| `perf_marker` | float | perfusion pressure (mmHg; lower = worse) |
| `temp_c` | float | core temperature, degrees Celsius |
| `acuity` | int 0..10 | bedside acuity score assigned by the care team at that decision point (higher = sicker) |
| `action` | int 0/1/2 | dose administered right after the observation: 0 = no drug, 1 = standard dose (1 unit), 2 = high dose (2 units) |
| `step_cost` | float | treatment cost charged for that action: 1.0 per dose unit |

The observations in a row were taken before the action in that row. The action
at `step` t influences the observations from `step` t+1 onwards and the final
outcome. Nothing is observed after the sixth action except the outcome.

## outcomes.csv (one row per patient)

| column | type | meaning |
|---|---|---|
| `patient_id` | int | joins to `patients.csv` |
| `recovery_score` | float 0..100 | recovery score measured one interval after the sixth action (higher = better) |
| `utility` | float | `recovery_score - sum(step_cost over the six steps)`; the quantity a policy should maximise in expectation |

## Clinical context

* The drug is an antimicrobial. How well it works against a patient's pathogen
  varies between patients; the baseline culture result is an imperfect
  predictor of that. It is eliminated from the body within roughly one
  treatment interval, so its effect does not carry over for long.
* Not every patient has an established infection at admission: some are only
  colonised and the infection may emerge (or never emerge) during the stay.
  A rising inflammation biomarker and temperature indicate an active infection.
* Uncontrolled infection damages organ function. So does the drug: its
  toxicity builds up with sustained exposure and is worse in frail patients and
  in patients whose organ function is already impaired.
* The bedside acuity score summarises how ill the patient looks; it rises with
  both infection and organ dysfunction and is the main input to the recorded
  dosing decisions.

## How the recorded care was delivered

* At every decision point the care team's dose choice depended only on
  information recorded in these files -- the baseline row, the observations at
  that and earlier decision points, and the earlier actions -- plus randomness
  independent of everything else. There are no unrecorded variables that
  influenced treatment choices.
* Treatment choice was stochastic: some patients received an uncommon action
  for their presentation, but several presentations almost never received a
  particular dose.
* The recorded care always respected the protocol in `protocol.json`.

## Protocol (`protocol.json`)

A dose sequence is feasible only if all of the following hold at every decision
point:

1. cumulative dose over the episode never exceeds 7 units;
2. a high dose (2 units) is never given when the current `renal_marker` exceeds 2.4;
3. a high dose is never given at the decision point immediately after a high dose.

Withholding the drug (action 0) is always feasible.
