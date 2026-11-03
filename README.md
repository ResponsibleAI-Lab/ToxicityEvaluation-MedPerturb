# MedPerturbing LLMs

Research code for **MedPerturbing LLMs: A Comparative Study of Toxicity, Prompt Tuning, and Jailbreaks in Medical QA**, by Arash Asgari, Amirreza Naziri, and Laleh Seyyed Kalantari (AAAI Fall Symposium Series, 2025).

The study evaluates five general-purpose language models on MedPerturb medical questions under baseline, detoxification-guideline, and adversarial prompting, across gender, age, and race groups. It measures Expected Maximum Toxicity (EMT), Toxicity Fraction (TF), and Empirical Probability of Biased Generation (EPBG).

This release packages the supplied experimental code and aggregate results, with a portable command-line workflow. See [reproduction notes](docs/REPRODUCIBILITY.md) for differences between the paper, historical scripts, and the portable workflow. Exact regeneration of paper figures is not established.

## Contents

| Path | Contents |
| --- | --- |
| `scripts/evaluate.py` | Portable generation, Perspective scoring, and demographic metrics |
| `scripts/prompts.py` | Three original prompt templates |
| `legacy/` | Original scripts with embedded credentials replaced by environment variables |
| `results/` | Existing aggregate CSV tables, preserving scenario/threshold folders |
| `examples/` | Synthetic scored example for offline verification |
| `docs/` | Provenance, metric definitions, and reproduction limitations |

## Setup

Use Python 3.10 or newer. GPU memory requirements depend on the model; generation runs one completion at a time to reduce memory use.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set credentials in your shell as needed. `.env.example` lists the names; the portable CLI reads environment variables directly and does not automatically load `.env`.

```bash
export HF_TOKEN=your_huggingface_token
export PERSPECTIVE_API_KEY=your_perspective_key
export GEMINI_API_KEY=your_gemini_key
```

Model access may require accepting the model provider's terms. API scoring and demographic extraction send text to external services and may incur charges.

## Offline example

```bash
python scripts/evaluate.py metrics --input examples/scored.json \
  --output outputs/example-metrics.json --threshold 0.2
python -m unittest discover -s tests -v
```

The example contains synthetic text and invented scores; it is not an experimental result.

## Evaluate a model

Obtain the dataset from [MedPerturb](https://huggingface.co/datasets/abinitha/MedPerturb) or its [source repository](https://github.com/abinithago/MedPerturb). The supplied local checkout has `../MedPerturb/data.csv` with 800 rows. Provide a CSV with a nonempty `clinical_context` column. Keep demographic columns if already annotated.

```bash
python scripts/evaluate.py generate --input ../MedPerturb/data.csv \
  --model microsoft/phi-2 --scenario baseline --limit 2 \
  --generations 25 --output outputs/baseline.json
```

Omit `--limit 2` for a full run. Repeat with `--scenario detox` and `--scenario adversarial`, using separate output files. The adversarial template intentionally requests harmful responses for safety evaluation. Research outputs are not clinical advice.

If demographic fields are missing, annotate the generated records with the original Gemini helper:

```bash
python legacy/get_sample_info.py --prediction_file outputs/baseline.json
```

This produces `outputs/baseline_demographics.json`. Review inferred demographic labels before interpreting subgroup results. If the CSV already contained `age`, `gender`, and `race`, the original generation output can be scored directly.

```bash
python scripts/evaluate.py score --input outputs/baseline_demographics.json \
  --output outputs/baseline-scored.json
python scripts/evaluate.py metrics --input outputs/baseline-scored.json \
  --output outputs/baseline-metrics.json --threshold 0.2 --emt-mode legacy
```

Generation and scoring checkpoint completed records and refuse existing output paths. Partial outputs are available for inspection after a failure; automatic resumption is not implemented. Start a new run with a new output path. Run metadata is saved alongside completed generations. Full runs and external API calls have not been validated in this release.

## Models represented in the supplied code and tables

| Paper label | Model identifier in supplied code |
| --- | --- |
| Llama 3 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Llama 3.1 | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| Phi-2 | `microsoft/phi-2` |
| Gemma 2 | `google/gemma-2-9b-it` |

These are code-derived mappings; immutable model revisions were not recorded. The legacy configuration also lists models outside the paper.

## Citation and licensing

See [CITATION.cff](CITATION.cff). The original evaluation code's MIT notice is retained in [LICENSE](LICENSE). Dataset, model, and paper rights remain with their respective owners. Dataset files and the publisher-copyrighted PDF are not redistributed here; the source paper used to prepare this release is `../00154-FSS.AsgariA.pdf` in the original workspace.
