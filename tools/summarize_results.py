"""Summarize the shared-reference protocol from preserved target-scene JSON records."""
from pathlib import Path
from statistics import mean
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=ROOT / 'results' / 'final')
args = parser.parse_args()
records = {}
for path in sorted(args.results.glob('*.json')):
    for key, value in json.loads(path.read_text(encoding='utf-8')).items():
        if key in records and records[key] != value:
            raise ValueError(f'Conflicting duplicate record: {key}')
        records[key] = value
scenes = ['ABU-A3', 'ABU-B3', 'ABU-U4', 'SanDiego', 'HYDICE', 'WHR']
protocols = [('P0', 'mean'), ('P1', 'pix1'), ('P2 (k=3)', 'pixk3'),
             ('P2 (k=5)', 'pixk5'), ('P4', 'perturb'), ('P3', 'cross')]
print('| Protocol | MetaMatch | MetaMatch-TTA | MetaMatch-TTA +S |')
print('| --- | ---: | ---: | ---: |')
for label, protocol in protocols:
    target_scenes = ['ABU-A3', 'ABU-U4', 'SanDiego'] if protocol == 'cross' else scenes
    draws = [0] if protocol in ('mean', 'cross') else [0, 1, 2]
    values = []
    for stage in ['zero', 'refined', 'final']:
        scene_means = []
        for scene in target_scenes:
            seed_means = []
            for seed in [0, 1, 2]:
                keys = [f'{scene}|{protocol}|{draw}|{seed}|{stage}' for draw in draws]
                missing = [key for key in keys if key not in records]
                if missing:
                    raise ValueError(f'Missing required records: {missing}')
                seed_means.append(mean(records[key]['AUC1'] for key in keys))
            scene_means.append(mean(seed_means))
        values.append(f'{mean(scene_means):.4f}')
    print('| ' + ' | '.join([label] + values) + ' |')
