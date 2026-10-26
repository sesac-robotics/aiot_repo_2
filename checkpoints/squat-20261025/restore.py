#!/usr/bin/env python3
"""Verify or restore the exact preserved files. Never starts a service/camera."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def entries(include_development):
    index = json.loads((BASE / 'INDEX.json').read_text())
    merged = {}
    snapshots = {}
    for group in ['runtime'] + (['development'] if include_development else []):
        for name, record in index[group].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe manifest path: ' + name)
            if name in merged and merged[name][1] != record:
                raise ValueError('Conflicting runtime/development files: ' + name)
            path = BASE / 'payload' / group / name
            snapshots[group + '/' + name] = (path, record)
            merged.setdefault(name, (path, record))
    return merged, snapshots


def verify_snapshot(records):
    for name, (path, record) in records.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing snapshot file: ' + name)
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Snapshot fingerprint mismatch: ' + name)


def verify_destination(destination, records):
    for name, (_, record) in records.items():
        path = destination / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing restored file: ' + name)
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Restored fingerprint mismatch: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-snapshot', action='store_true')
    modes.add_argument('--destination', type=Path, help='Restore into a new or empty folder only')
    modes.add_argument('--verify-only', type=Path, help='Read-only comparison against an existing folder')
    parser.add_argument('--include-development', action='store_true', help='Also restore the explicitly undeployed model-development artifacts')
    args = parser.parse_args()
    records, snapshots = entries(args.include_development)
    verify_snapshot(snapshots)
    if args.destination is not None:
        destination = args.destination.expanduser().absolute()
        if destination.is_symlink() or destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            parser.error('Destination must be a new or empty folder; no existing work is overwritten.')
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        for name, (source, _) in records.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        verify_destination(destination, records)
        action = 'restored_and_all_hashes_verified'
    elif args.verify_only is not None:
        verify_destination(args.verify_only.expanduser().resolve(), records)
        action = 'existing_files_verified_read_only'
    else:
        action = 'snapshot_verified'
    print(json.dumps({'action': action, 'files_verified': len(records), 'snapshot_files_verified': len(snapshots), 'includes_undeployed_development': args.include_development, 'service_started': False, 'camera_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
