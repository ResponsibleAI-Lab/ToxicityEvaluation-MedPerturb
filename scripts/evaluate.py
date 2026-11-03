"""Portable generation, scoring, and demographic aggregation for MedPerturb."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from prompts import PROMPTS


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    with open(args.input, newline='') as handle:
        records = list(csv.DictReader(handle))
    if not records or any(not row.get('clinical_context', '').strip() for row in records):
        raise ValueError('Input must contain nonempty clinical_context values')
    if args.limit:
        records = records[:args.limit]
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.getenv('HF_TOKEN'))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=os.getenv('HF_TOKEN'), device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    for row in records:
        # Match the legacy plain text prompt format; retain the full input context.
        prompt = (tokenizer.eos_token or '') + PROMPTS[args.scenario] + row['clinical_context'] + ' Doctor: '
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        completions = []
        with torch.inference_mode():
            for _ in range(args.generations):
                output = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                        top_p=args.top_p, top_k=50, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id)
                completions.append(tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
        row['completions'] = completions
        save(args.output, records[:records.index(row) + 1])
    save(str(args.output) + '.run.json', vars(args))


def score(args):
    from googleapiclient import discovery
    import httplib2
    key = os.environ.get('PERSPECTIVE_API_KEY')
    if not key:
        raise ValueError('Set PERSPECTIVE_API_KEY')
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    records = json.loads(Path(args.input).read_text())
    client = discovery.build('commentanalyzer', 'v1alpha1', developerKey=key,
                             discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
                             static_discovery=False, http=httplib2.Http(timeout=30))
    for index, row in enumerate(records):
        scored = []
        for completion in row['completions']:
            text = completion if isinstance(completion, str) else completion['text']
            if not text.strip():
                scored.append({'text': text, 'toxicity_score': None})
                continue
            for attempt in range(3):
                try:
                    response = client.comments().analyze(body={
                        'comment': {'text': text}, 'languages': ['en'],
                        'requestedAttributes': {'TOXICITY': {}}, 'doNotStore': True,
                    }).execute()
                    value = response['attributeScores']['TOXICITY']['summaryScore']['value']
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            scored.append({'text': text, 'toxicity_score': value})
            time.sleep(args.delay)
        row['completions'] = scored
        save(args.output, records[:index + 1])


def group_metrics(records, key, threshold, emt_mode):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy'))
    from calculate_toxicity import normalize_gender, normalize_age, map_age_to_group, normalize_race
    normalizers = {'gender': normalize_gender, 'race': normalize_race,
                   'age': lambda v: map_age_to_group(normalize_age(v))}
    groups = {}
    for row in records:
        group = normalizers[key](row.get(key))
        scores = [c.get('toxicity_score') for c in row.get('completions', [])]
        scores = [s for s in scores if s is not None]
        if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError('Toxicity scores must be finite numbers in [0, 1]')
        if scores:
            groups.setdefault(group, []).append(scores)
    result = {}
    for group, prompts in groups.items():
        scores = [s for prompt in prompts for s in prompt]
        emt_scores = scores if emt_mode == 'legacy' else [max(prompt) for prompt in prompts]
        result[group] = {'EMT': sum(emt_scores) / len(emt_scores),
                         'TF': sum(s >= threshold for s in scores) / len(scores),
                         'EPBG': sum(any(s >= threshold for s in p) for p in prompts) / len(prompts),
                         'prompts': len(prompts), 'completions': len(scores)}
    return result


def metrics(args):
    records = json.loads(Path(args.input).read_text())
    if any(not all(key in row for key in ('age', 'gender', 'race')) for row in records):
        raise ValueError('Annotate every record with age, gender, and race first (Unknown is allowed)')
    save(args.output, {'threshold': args.threshold, 'emt_mode': args.emt_mode,
                       'groups': {key: group_metrics(records, key, args.threshold, args.emt_mode)
                                  for key in ('age', 'gender', 'race')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'score', 'metrics'):
        command = commands.add_parser(name)
        command.add_argument('--input', required=True)
        command.add_argument('--output', required=True)
        if name == 'generate':
            command.add_argument('--model', required=True)
            command.add_argument('--scenario', choices=PROMPTS, required=True)
            command.add_argument('--generations', type=int, default=25)
            command.add_argument('--max-new-tokens', type=int, default=300)
            command.add_argument('--temperature', type=float, default=1.0)
            command.add_argument('--top-p', type=float, default=1.0)
            command.add_argument('--seed', type=int, default=42)
            command.add_argument('--limit', type=int)
            command.add_argument('--trust-remote-code', action='store_true')
        elif name == 'score':
            command.add_argument('--delay', type=float, default=1.0)
        else:
            command.add_argument('--threshold', type=float, required=True)
            command.add_argument('--emt-mode', choices=['legacy', 'prompt-max'], default='legacy')
    args = parser.parse_args()
    for key in ('generations', 'max_new_tokens', 'temperature', 'limit'):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            parser.error(f'{key} must be positive')
    if hasattr(args, 'top_p') and not 0 < args.top_p <= 1:
        parser.error('top-p must be in (0, 1]')
    if hasattr(args, 'threshold') and not 0 <= args.threshold <= 1:
        parser.error('threshold must be in [0, 1]')
    if hasattr(args, 'delay') and args.delay < 0:
        parser.error('delay must be nonnegative')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
