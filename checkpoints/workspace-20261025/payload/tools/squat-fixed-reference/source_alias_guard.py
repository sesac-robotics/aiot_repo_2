"""Content identity guard for reviewed source aliases; never reassign a split."""
from collections import defaultdict


def index_sources(records):
    groups = defaultdict(list)
    for row in records:
        digest = row['source_sha256']
        if not digest or row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Missing content identity or canonical role')
        groups[digest].append(row)
    return dict(groups)


def assert_alias_safe_training(rows, records):
    """A filename marked Train cannot override held-out byte-identical content."""
    index = index_sources(records)
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('Non-Train row: ' + row['id'])
        hashes = row.get('sources', [])
        if isinstance(hashes, str):
            hashes = [hashes]
        hashes = set(hashes) | {row.get('source_sha256')}
        for digest in hashes:
            roles = {r['split'] for r in index.get(digest, [])}
            if roles - {'train'}:
                raise ValueError('Held-out content alias: ' + row['id'])


def assert_disjoint_content(fit_rows, held_rows):
    def hashes(rows):
        result = set()
        for row in rows:
            values = row.get('sources', [])
            if isinstance(values, str):
                values = [values]
            result.update(values)
            if row.get('source_sha256'):
                result.add(row['source_sha256'])
        return result
    if hashes(fit_rows) & hashes(held_rows):
        raise ValueError('Byte-identical source crosses fit/held boundary')
