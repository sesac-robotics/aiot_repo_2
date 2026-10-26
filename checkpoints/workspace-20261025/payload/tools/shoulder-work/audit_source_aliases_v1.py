from pathlib import Path
import json,hashlib,collections,datetime
B=Path('/home/jayden/SeSACPAI-Mid');R=B/'output/shoulder-work-20261024/full-review-v1';O=B/'output/shoulder-work-20261024/source-alias-audit-v1-20261025';O.mkdir(exist_ok=True)
inv=json.loads((R/'inventory.frozen.json').read_text())['sources'];groups=collections.defaultdict(list);mismatch=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
for r in inv:
 actual=sha(r['path'])
 if actual!=r['source_sha256']:mismatch.append(r['id'])
 groups[actual].append(r)
assert not mismatch
aliases=[]
for digest,ss in groups.items():
 if len(ss)<2:continue
 roles=sorted({s['split'] for s in ss});views=sorted({s['view'] for s in ss})
 aliases.append(dict(source_sha256=digest,canonical_source_group='shoulder_sha256_'+digest,
     aliases=[{k:s[k] for k in ['id','path','person','recording_group','view','provider_tag','split']} for s in ss],
     cross_split=len(roles)>1,roles=roles,view_metadata_conflict=len(views)>1,
     provider_tag_conflict=len({s['provider_tag'] for s in ss})>1,
     future_use='one source group; do not duplicate fit weight or cross group CV; hold conflicting view alias until visual bank reviewed'))
report=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=len(inv),
    unique_source_sha256=len(groups),duplicate_groups=len(aliases),duplicate_aliases=sum(len(a['aliases'])-1 for a in aliases),
    all_original_source_hashes_recomputed=True,source_hash_mismatches=mismatch,
    file_role_counts=dict(collections.Counter(s['split'] for s in inv)),
    unique_source_role_counts=dict(collections.Counter(next(iter({s['split'] for s in ss})) for ss in groups.values() if len({s['split'] for s in ss})==1)),
    cross_role_duplicate_groups=sum(a['cross_split'] for a in aliases),aliases=aliases,
    source_roles_changed=False,models_changed=False,fit_performed=False,Test_content_opened=False,
    limitations=['hash audit only; independent accuracy not measured','150 source paths are 147 unique byte-identical videos','near-duplicate encodings are not covered'])
p=O/'source-aliases.frozen.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='aliases'},ensure_ascii=False))
# Preserve authoritative grouping for future candidate training without editing old labels or splits.
rows=[]
for digest,ss in groups.items():
 for s in ss:
  rows.append(dict(id=s['id'],source_sha256=digest,person=s['person'],original_recording_group=s['recording_group'],
      source_group='shoulder_sha256_'+digest,split=s['split'],view=s['view'],provider_tag=s['provider_tag'],
      alias_count=len(ss),metadata_view_conflict=len({v['view'] for v in ss})>1,
      training_role_allowed=s['split']=='train',new_label_assigned=False))
(O/'source-groups.frozen.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
