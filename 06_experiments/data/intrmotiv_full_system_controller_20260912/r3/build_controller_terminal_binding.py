from pathlib import Path
import subprocess,sysconfig,numpy,shutil,hashlib,json,difflib
source=Path('/home/fr/fr_xl1014/deepmindlab/lab')
base=Path('/home/fr/fr_xl1014/.conda/envs/SFgit/lib/python3.10/site-packages/deepmind_lab')
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1')
pkg=root/'deepmind_lab';pkg.mkdir(parents=True,exist_ok=True)
original=(source/'python/dmlab_module.c').read_text()
old='static PyObject* Lab_observations(PyObject* pself, PyObject* no_arg) {'
assert original.count(old)==1
patched=original.replace(old,'static PyObject* Lab_read_observations(PyObject* pself, int terminal) {')
old='  if (!is_running((self))) {'
assert patched.count(old)==1
patched=patched.replace(old,'  if (terminal ? self->status != EnvCApi_EnvironmentStatus_Terminated : !is_running(self)) {')
needle='static PyObject* Lab_events(PyObject* pself, PyObject* no_arg) {'
patched=patched.replace(needle,'''// Explicit opt-in final-state access; normal observations() retains its guard.
static PyObject* Lab_observations(PyObject* pself, PyObject* no_arg) {
  return Lab_read_observations(pself, 0);
}

static PyObject* Lab_terminal_observations(PyObject* pself, PyObject* no_arg) {
  return Lab_read_observations(pself, 1);
}

'''+needle)
needle='    {"observations", Lab_observations, METH_NOARGS, "Get the observations"},'
assert patched.count(needle)==1
patched=patched.replace(needle,needle+'\n    {"terminal_observations", Lab_terminal_observations, METH_NOARGS, "Render final engine state before reset; only valid after termination"},')
(root/'dmlab_module.c').write_text(patched)
(root/'terminal_binding.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='a/python/dmlab_module.c',tofile='b/python/dmlab_module.c')))
for item in base.iterdir():
 if item.name in ('deepmind_lab.so','__pycache__'):continue
 dest=pkg/item.name
 if not dest.exists():dest.symlink_to(item)
# Link the new Python binding to the already installed engine loader. Its code,
# engine DSOs and assets remain byte-for-byte unchanged; no global install edits.
loader=pkg/'libdmlab_binding_base.so'
if not loader.exists():shutil.copy2(base/'deepmind_lab.so',loader)
cmd=['gcc','-shared','-fPIC','-O2','-std=c99','-fno-strict-aliasing','-DDEEPMIND_LAB_MODULE_RUNFILES_DIR',
 '-I'+str(source),'-I'+numpy.get_include(),'-I'+sysconfig.get_path('include'),str(root/'dmlab_module.c'),
 '-L'+str(pkg),'-Wl,-rpath,$ORIGIN','-l:libdmlab_binding_base.so','-o',str(pkg/'deepmind_lab.so')]
subprocess.run(cmd,check=True)
manifest=dict(schema='intrmotiv/terminal-binding/v1',source_sha256=hashlib.sha256(original.encode()).hexdigest(),
 binding_sha256=hashlib.sha256((pkg/'deepmind_lab.so').read_bytes()).hexdigest(),
 original_loader_sha256=hashlib.sha256(loader.read_bytes()).hexdigest(),overlay=str(root),build_command=cmd)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
