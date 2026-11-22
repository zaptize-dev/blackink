#!/usr/bin/env python3
"""Stage the Hugging Face release in one directory, verify it, then (with --publish) upload it.

The Hub repo is the complete ready-to-run model: the unchanged Inkling-2-Mini files plus the
ablation file, with card, licenses, checksums and a provenance folder:
  config.json ... mtp.safetensors, tiktoken/   unchanged Inkling-2-Mini files
  inkling-2-mini-abliterated.safetensors       the ablation
  README.md, LICENSE, NOTICE, SHA256SUMS       card, licenses, checksums
  abliteration/                                tools and results/ (provenance)

Staging is idempotent: big files are reflinked (cp --reflink=auto, hard-link fallback), files no
longer in the allowlist are removed from the stage. Publishing mirrors the stage: Hub files the
stage lacks are deleted, then every file's size and every LFS file's sha256 is verified remotely.

    tools/.venv/bin/python tools/publish_hf.py                         # stage + verify only
    HF_TOKEN=... tools/.venv/bin/python tools/publish_hf.py --publish  # stage + verify + upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = 'satellitedown/Inkling-2-Mini-abliterated-ingine'
MODEL_FILES = [
    'config.json', 'generation_config.json', 'hf_quant_config.json', 'chat_template.jinja',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'processor_config.json',
    'model.safetensors.index.json', 'model-00001-of-00001.safetensors', 'mtp.safetensors',
]
MODEL_DIRS = ['tiktoken']
REPO_FILES = {  # stage path -> project path
    'inkling-2-mini-abliterated.safetensors': 'ablation/inkling-2-mini-abliterated.safetensors',
    'README.md': 'hf/README.md',
    'LICENSE': 'LICENSE',
    'NOTICE': 'NOTICE',
}
PROVENANCE = ['ablate.py', 'inkling_ref.py', 'refusal_eval.py', 'capability_eval.py', 'reasoning_eval.py', 'eval.sh',
              'requirements.txt']
CLEAN_PATHS = ['hf/README.md', 'LICENSE', 'NOTICE', 'ablation', 'tools', 'results']
BIG = 64 << 20


def die(msg: str) -> None:
    print(f'error: {msg}', file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    """Copy src to dst unless an identical-looking copy is already there."""
    st = src.stat()
    if dst.is_file():
        d = dst.stat()
        if d.st_size == st.st_size and int(d.st_mtime) == int(st.st_mtime):
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if st.st_size >= BIG:
        r = subprocess.run(['cp', '--reflink=auto', '--preserve=timestamps', str(src), str(dst)])
        if r.returncode != 0:
            dst.unlink(missing_ok=True)
            os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def plan(model_dir: Path) -> dict[str, Path]:
    """Allowlist: stage-relative path -> source file."""
    files: dict[str, Path] = {}
    missing: list[str] = []
    for name in MODEL_FILES:
        p = model_dir / name
        (files.__setitem__(name, p) if p.is_file() else missing.append(str(p)))
    for d in MODEL_DIRS:
        p = model_dir / d
        found = sorted(f for f in p.rglob('*') if f.is_file()) if p.is_dir() else []
        if not found:
            missing.append(f'{p}/ (no files)')
        for f in found:
            files[f.relative_to(model_dir).as_posix()] = f
    for dst, src in REPO_FILES.items():
        p = PROJECT / src
        (files.__setitem__(dst, p) if p.is_file() else missing.append(str(p)))
    for name in PROVENANCE:
        p = PROJECT / 'tools' / name
        (files.__setitem__(f'abliteration/{name}', p) if p.is_file() else missing.append(str(p)))
    results = sorted((PROJECT / 'results').glob('*.json'))
    if not results:
        missing.append(f'{PROJECT / "results"}/*.json')
    for p in results:
        files[f'abliteration/results/{p.name}'] = p
    if missing:
        die('missing release files:\n  ' + '\n  '.join(missing))
    return files


def staged_files(stage: Path) -> list[str]:
    return sorted(f.relative_to(stage).as_posix() for f in stage.rglob('*')
                  if f.is_file() and f.relative_to(stage).parts[0] != '.cache')


def do_stage(model_dir: Path, stage: Path) -> dict[str, str]:
    files = plan(model_dir)
    stage.mkdir(parents=True, exist_ok=True)
    print(f'staging {len(files)} files from {model_dir} and {PROJECT} into {stage}', flush=True)
    for rel, src in files.items():
        place(src, stage / rel)
    keep = set(files) | {'SHA256SUMS'}
    for rel in staged_files(stage):
        if rel not in keep:
            print(f'  removing stale {rel}')
            (stage / rel).unlink()
    for d in sorted((p for p in stage.rglob('*') if p.is_dir() and '.cache' not in p.relative_to(stage).parts),
                    key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()

    print('hashing staged files and verifying against sources', flush=True)
    sums: dict[str, str] = {}
    for rel in sorted(files):
        s = sha256(stage / rel)
        src = files[rel]
        if (stage / rel).stat().st_size != src.stat().st_size or (
                src.stat().st_size < BIG and sha256(src) != s):
            die(f'staged {rel} differs from {src}')
        sums[rel] = s
    (stage / 'SHA256SUMS').write_text(''.join(f'{sums[r]}  {r}\n' for r in sorted(sums)))
    sums['SHA256SUMS'] = sha256(stage / 'SHA256SUMS')

    total = 0
    for rel in staged_files(stage):
        size = (stage / rel).stat().st_size
        total += size
        print(f'  {size:>16,}  {rel}')
    print(f'staged {len(sums)} files, {total / 2**30:.2f} GiB in {stage}')
    if set(staged_files(stage)) != set(sums):
        die('stage contents do not match the allowlist')
    return sums


def check_clean() -> None:
    r = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal', '--', *CLEAN_PATHS],
                       cwd=PROJECT, capture_output=True, text=True)
    if r.returncode != 0:
        die(f'git status failed: {r.stderr.strip()}')
    if r.stdout.strip():
        print(r.stdout, end='', file=sys.stderr)
        die('uncommitted changes in release sources; commit them or pass --allow-dirty')


def publish(stage: Path, repo_id: str, private: bool, sums: dict[str, str]) -> None:
    from huggingface_hub import CommitOperationDelete, HfApi
    if not os.environ.get('HF_TOKEN'):
        die('HF_TOKEN is not set')
    api = HfApi(token=os.environ['HF_TOKEN'])
    owner = repo_id.split('/')[0]
    user = api.whoami()['name']
    if user != owner:
        die(f'authenticated as {user!r}, but the repo owner is {owner!r}')

    api.create_repo(repo_id, repo_type='model', private=private, exist_ok=True)
    print(f'uploading {stage} -> https://huggingface.co/{repo_id} (unchanged files are skipped)', flush=True)
    api.upload_folder(repo_id=repo_id, repo_type='model', folder_path=str(stage),
                      commit_message='Upload Inkling-2-Mini abliterated release',
                      ignore_patterns=['.cache/**'])

    staged = set(staged_files(stage))
    stale = sorted(set(api.list_repo_files(repo_id, repo_type='model')) - staged - {'.gitattributes'})
    if stale:
        api.create_commit(repo_id, repo_type='model',
                          operations=[CommitOperationDelete(path_in_repo=p) for p in stale],
                          commit_message=f'Remove {len(stale)} files no longer in the release')
        print(f'removed {len(stale)} stale Hub files: {", ".join(stale)}', flush=True)

    print('verifying remote files', flush=True)
    info = api.model_info(repo_id, revision='main', files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings or []}
    errors: list[str] = []
    report: dict[str, dict] = {}
    for rel in sorted(staged):
        size = (stage / rel).stat().st_size
        s = remote.get(rel)
        if s is None:
            errors.append(f'{rel}: missing on Hub')
            continue
        if s.size != size:
            errors.append(f'{rel}: size {s.size} on Hub, {size} staged')
        if s.lfs is not None and s.lfs.sha256 != sums[rel]:
            errors.append(f'{rel}: LFS sha256 {s.lfs.sha256} on Hub, {sums[rel]} staged')
        report[rel] = {'sha256': sums[rel], 'size': size}
    extra = sorted(set(remote) - staged - {'.gitattributes'})
    errors += [f'{p}: on Hub but not staged' for p in extra]
    if errors:
        die('remote verification failed:\n  ' + '\n  '.join(errors))

    out = PROJECT / 'reports' / 'huggingface-publish.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'repo': repo_id, 'revision': info.sha, 'files': report}, indent=1) + '\n')
    print(f'verified {len(report)} files; wrote {out}')
    print(f'done: https://huggingface.co/{repo_id} @ {info.sha}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', type=Path,
                    default=Path(os.environ.get('INGINE_MODEL_DIR') or PROJECT / 'models' / 'Inkling-2-Mini'),
                    help='Inkling-2-Mini model dir (default: $INGINE_MODEL_DIR or <repo>/models/Inkling-2-Mini)')
    ap.add_argument('--stage', type=Path, default=PROJECT / 'work' / 'hf-release',
                    help='staging dir; put it on the model\'s filesystem so the weights are reflinked, not copied '
                         '(default: %(default)s)')
    ap.add_argument('--repo', default=DEFAULT_REPO, help='Hub repo id (default: %(default)s)')
    ap.add_argument('--private', action='store_true', help='create the repo private (default: public)')
    ap.add_argument('--publish', action='store_true', help='upload; without it only stage and verify')
    ap.add_argument('--allow-dirty', action='store_true', help='allow uncommitted changes in release sources')
    a = ap.parse_args()

    if a.publish and not a.allow_dirty:
        check_clean()
    sums = do_stage(a.model_dir.resolve(), a.stage)
    if a.publish:
        publish(a.stage, a.repo, a.private, sums)
    else:
        print('stage only; rerun with --publish to upload')
    return 0


if __name__ == '__main__':
    sys.exit(main())
