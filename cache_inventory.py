"""Read existing confidence caches; write inventory only, with no fitting/extraction."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch

PROJECT = Path('/Users/cril/tanmoy/research/Deferal_Models/LearningtoCalibrate-aistats-2027')
CACHE_ROOT = PROJECT / 'letter11_provenance_v1'
MODELS = ['llama3_8b', 'mistral_7b', 'qwen2_5_7b']
DATASETS = ['pavlick_nli', 'truthfulqa']
OUT = Path(__file__).resolve().parent / 'arr_results' / 'cache_inventory'

def inventory(model, dataset):
    cache = CACHE_ROOT / model / dataset
    errors, n = [], None
    for arm in ['forward', 'reversed']:
        path = cache / (arm + '.pt')
        if not path.exists():
            errors.append(f'missing {arm}')
            continue
        z = torch.load(path, map_location='cpu', weights_only=False)
        meta = z['meta']
        values = np.asarray(z['conf_values'], dtype=float)
        if values.max() > 1:
            values = values / 100
        for suffix in ['pos', 'neg']:
            dist = np.asarray(z['Vdist_' + suffix], dtype=float)
            v = np.asarray(z['V_' + suffix], dtype=float)
            if not np.allclose(dist @ values, v, atol=1e-6):
                errors.append(f'{arm}/{suffix}: expectation mismatch')
            if not np.allclose(dist.sum(axis=1), 1, atol=1e-5):
                errors.append(f'{arm}/{suffix}: normalization mismatch')
        if meta.get('legacy_artifact_used') is not False or not meta.get('model_revision'):
            errors.append(f'{arm}: insufficient explicit provenance')
        n = len(z['example_ids'])
    return dict(model=model, dataset=dataset, cache_exists=cache.exists(),
                reproducible=not errors, reproduced_quantity='V = sum(conf_values * Vdist), both arms/candidates',
                provenance_scope='Pinned metadata and saved-distribution replay; not an extraction rerun',
                n_questions=n, seen_before_estimand_note='unsure',
                bears_on_theory=None,
                reason='No cell-count sweep or pointwise/cell-level coverage functional in these report caches',
                errors=errors, classification='not a theory demonstration')

rows = [inventory(m, d) for m in MODELS for d in DATASETS]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'inventory.json').write_text(json.dumps(rows, indent=2) + '\n')
with (OUT / 'inventory.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# Separate existing pilot: replay saved optimal-M numbers, not a new experiment.
pilot = PROJECT / 'mondrian/runs/popqa_seed0_group_scaling'
manifest = json.loads((pilot / 'manifest.json').read_text())
hash_checks = {name: hashlib.sha256((pilot/name).read_bytes()).hexdigest() == digest
               for name, digest in manifest['output_sha256'].items() if (pilot/name).exists()}
z = np.load(pilot / 'losses.npz', allow_pickle=False)
means = z['loss'].mean(axis=-1)
published = list(csv.DictReader((pilot / 'M_star.csv').open()))
replay = []
for gi, goal in enumerate(z['goals']):
    for ni, ncal in enumerate(z['n_cal_grid']):
        best = int(z['M_grid'][np.argmin(means[gi, ni])])
        row = next(r for r in published if r['goal'] == str(goal) and int(r['n_cal']) == int(ncal))
        replay.append(dict(goal=str(goal), n_cal=int(ncal), M_star=best,
                           matches_saved=best == int(row['M_star'])))
(OUT / 'existing_pilot_replay.json').write_text(json.dumps(dict(
    cache=str(pilot), output_hash_checks=hash_checks, replay=replay,
    classification='exploratory illustrative pilot; not a confirmatory test of pointwise theory',
    plan_created_utc=json.loads((pilot/'plan.json').read_text())['created_utc'],
    estimand_note='Referenced 2026-10-02 note is absent from this local reports directory'), indent=2)+'\n')
lines = ['| Model | Dataset | Exists | Report replay | Seen before note | Theory link |',
         '|---|---|---|---|---|---|']
lines += [f"| {r['model']} | {r['dataset']} | {r['cache_exists']} | {r['reproducible']} | unsure | none established |" for r in rows]
(OUT / 'APPENDIX.md').write_text('\n'.join(lines)+'\n\nReport replay verifies the saved confidence expectation, not calibration or a theory prediction. Unknown chronology is not counted as unseen.\n')
print(json.dumps(dict(rows=len(rows), report_replay_pass=sum(r['reproducible'] for r in rows),
    strict_body_survivors=sum(r['reproducible'] and r['seen_before_estimand_note'] is False and r['bears_on_theory'] is not None for r in rows),
    pilot_replay_pass=all(r['matches_saved'] for r in replay), pilot_hash_pass=all(hash_checks.values()), pilot=replay), indent=2))
