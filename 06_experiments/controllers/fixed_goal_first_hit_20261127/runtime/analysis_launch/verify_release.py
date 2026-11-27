"""Verify the reviewed analysis source and launcher manifest before/after work."""
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Analysis manifest changed'
d=json.loads(p.read_text());root=pathlib.Path(d['source_release']);launch=pathlib.Path(d['launch_directory'])
for relative,sha in d['source_files_sha256'].items():
 path=root/relative;assert path.is_file() and not path.is_symlink()
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,relative
for relative,sha in d['launcher_helpers_sha256'].items():
 assert hashlib.sha256((launch/relative).read_bytes()).hexdigest()==sha,relative
print(json.dumps({'source_commit':d['source_commit'],'source_files_verified':len(d['source_files_sha256']),'source_unchanged':True}))
