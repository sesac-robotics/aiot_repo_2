"""Protect the previously held-out team participant from training reuse."""
import json
from pathlib import Path

ROOT = Path('/home/jayden/SeSACPAI-Mid')
POLICY = ROOT / 'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'

def assert_training_allowed(rows):
    policy = json.loads(POLICY.read_text())
    blocked_ids = set(policy['protected_clip_ids'])
    blocked_hashes = set(policy['protected_source_sha256'])
    for row in rows:
        person = str(row.get('person', '')).removeprefix('r2d2_')
        hashes = row.get('sources', [])
        if isinstance(hashes, str):
            hashes = [hashes]
        hashes = set(hashes) | {row.get('source_sha256'), row.get('source')}
        if (person in policy.get('protected_people', []) or row.get('id') in blocked_ids
                or hashes & blocked_hashes):
            raise ValueError('Protected team holdout cannot train: ' + str(row.get('id')))
        if row.get('split', 'train') != 'train':
            raise ValueError('Non-training row passed to fit: ' + str(row.get('id')))
