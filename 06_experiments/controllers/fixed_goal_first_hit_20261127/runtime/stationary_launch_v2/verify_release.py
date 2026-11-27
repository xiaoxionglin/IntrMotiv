"""Check exact source inventory and reviewed immutable input bytes."""
import hashlib,json,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
manifest_path=pathlib.Path(sys.argv[1]);assert sha(manifest_path)==sys.argv[2]
m=json.loads(manifest_path.read_text());root=pathlib.Path(m['source_release'])
actual={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
assert actual==m['files_sha256'],'Immutable candidate source inventory changed'
for path,expected in m['qualification_input_files_sha256'].items():assert sha(path)==expected,path
print(json.dumps({'source_files_unchanged':len(actual),'qualification_input_files_unchanged':len(m['qualification_input_files_sha256']),'native_model_files_unchanged':len(m['model_files_sha256'])}))
