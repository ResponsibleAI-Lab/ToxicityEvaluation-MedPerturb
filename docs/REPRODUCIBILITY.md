# Reproduction and provenance

This repository was assembled from `00154-FSS.AsgariA.pdf`, `Toxicity-Fairness-Metrics/`, and the dataset description/schema in the local `MedPerturb/` checkout. Original evaluation script hashes appear in `source-manifest.json`. Legacy scripts retain their experimental behavior except credential literals, which now read environment variables. They may have absolute paths, import-time execution, and incomplete exploratory routines; use the portable CLI for new runs.

## Metric differences

The paper describes EMT as a mean of per-prompt maxima and TF as token-level toxicity. The supplied `calculate_toxicity.py` instead implements:

- EMT: mean of all valid completion toxicity scores.
- TF: fraction of valid completions with score greater than or equal to the threshold.
- EPBG: fraction of prompts with at least one valid completion at or above the threshold.

The portable CLI defaults to these legacy calculations. `--emt-mode prompt-max` changes EMT to the mean of per-prompt maximum scores. Neither mode implements token-level TF; Perspective summary scores are response-level. Metric output records the mode, threshold, prompt counts, and valid completion counts. Null scores are excluded; prompts without valid scores are excluded. Missing demographic labels normalize to `unknown` using the legacy mappings. These mappings are broad research groupings and do not establish a person's identity.

Legacy bootstrap intervals use 1,000 resamples of 80% of prompts with replacement, without a fixed random seed. Stored means are bootstrap means. The portable CLI reports direct point estimates and does not generate bootstrap intervals. The legacy metric script only discovers filenames ending `_processed.json` and hardcodes threshold 0.2; it cannot automatically reproduce all archived threshold folders.

## Generation and scoring differences

Original defaults include 25 generations, temperature 1.0, top-p 1.0, implicit top-k 50, and a 300-token length setting in the runner. The original generator selects the adversarial prompt in source, truncates inputs to `max_len`, decodes input plus output, and sets total output length from an untruncated baseline prompt length. The old runner also deletes model cache folders after generation.

The portable workflow explicitly selects a scenario, keeps full input text, generates up to 300 **new** tokens, and scores only generated response text. It does not strip a longest common prefix across completions. It seeds generation and records arguments. These corrections mean new scores need not match historical scores. The original seed, model revisions, software lockfile, exact scoring service version, and complete run provenance are unavailable. Deterministic agreement across GPU environments is not guaranteed.

## Archived results

`results/` contains aggregate CSVs copied byte-for-byte from `Toxicity-Fairness-Metrics/Toxicity_results/`. Folder names are retained, including `normal prompt`, `debias prompt`, `toxic prompt`, and threshold variants. A folder or filename is historical metadata, not proof that a table generated a specific published panel. No panel-to-table mapping has been inferred. Raw patient text, completion dumps, upstream Git history, environment files, and checkpoints are excluded.

`results-manifest.json` records relative paths and SHA-256 hashes for all released tables. Existing charts and notebooks are omitted to keep the release focused and avoid publishing hidden notebook state. Original plotting scripts are retained in `legacy/` but contain hardcoded paths and are not wired to the new JSON metrics format.

## Local verification

The offline tests distinguish legacy EMT from per-prompt-max EMT, check threshold equality, exclude missing scores, and validate grouping. No model downloads, Gemini annotation, or Perspective requests are needed. End-to-end model/API experiments require credentials and compute and were not run during packaging.

## Credentials

The original workspace contains hardcoded API credentials. Copies in this release use environment variables, and `.gitignore` excludes environment files. Rotate the exposed source credentials before publishing the original folders or histories. This release does not import their Git history.
