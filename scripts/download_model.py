#!/usr/bin/env python3
"""Download and verify the pinned Inkling-2-Mini abliterated model for ingine.

Fetches exactly the files listed in runtime-manifest.json at the pinned revision of the model repo
(base checkpoint, MTP drafter, tokenizer and the ablation file) and checks each one's size and SHA-256
against the manifest. Interrupted downloads resume. Files that do not match are moved to
<model folder>.previous-<time>/, never deleted, and downloaded again.

Target folder: --model-dir, else <--models-dir>/Inkling-2-Mini, else $INGINE_MODEL_DIR, else
<repo>/models/Inkling-2-Mini. Exit status: 0 verified, 1 missing/mismatching files or download error.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Free space required beyond the files themselves: download metadata and filesystem overhead.
SPACE_MARGIN = 1 << 30


def load_model(manifest_path):
    model = json.loads(manifest_path.read_text(encoding="utf-8"))["model"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repo_id"]):
        raise ValueError("Invalid model repo_id in runtime-manifest.json")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]) or not model["files"]:
        raise ValueError("The installer is not pinned yet: runtime-manifest.json needs the model's full commit hash "
                         "and file checksums (maintainers: python tools/pin.py). Update this checkout and retry.")
    for name in (model["directory"], *model["files"]):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise ValueError(f"Expected a relative path in runtime-manifest.json: {name}")
    for name, pin in model["files"].items():
        if not (re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) and isinstance(pin["size"], int)):
            raise ValueError(f"Invalid pin for {name} in runtime-manifest.json")
    return model


def human(size):
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(destination, files, names):
    """Map each name to None (matches its pin) or a reason it does not."""
    def one(name):
        path, pin = destination / name, files[name]
        if not path.exists():
            return "missing"
        if not path.is_file():
            return "not a regular file"
        size = path.stat().st_size
        if size != pin["size"]:
            return f"size {size}, expected {pin['size']}"
        digest = file_digest(path)
        return None if digest == pin["sha256"] else f"SHA-256 {digest}, expected {pin['sha256']}"

    # Hashing releases the GIL, so large files are checked in parallel.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        return dict(zip(names, pool.map(one, names)))


def previous_directory(destination):
    """Create a new sibling folder for files that do not match the pins."""
    stem = f"{destination.name}.previous-{time.strftime('%Y%m%d-%H%M%S')}"
    for attempt in range(1, 100):
        candidate = destination.with_name(stem if attempt == 1 else f"{stem}-{attempt}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise ValueError(f"Could not create a folder for the previous files next to {destination}")


def set_aside(path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.rename(target)
    except OSError as error:
        raise ValueError(f"Could not move {path} to {target}: {error}\n"
                         "Move it out of the model folder yourself and retry.") from error


def target_directory(args, model):
    if args.model_dir:
        return args.model_dir
    if args.models_dir:
        return args.models_dir / model["directory"]
    if os.environ.get("INGINE_MODEL_DIR"):
        return Path(os.environ["INGINE_MODEL_DIR"])
    return ROOT / "models" / model["directory"]


def report(results):
    for name, problem in results.items():
        print(f"Verified: {name}" if problem is None else f"Mismatch: {name}: {problem}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models-dir", type=Path, help="parent folder; the model goes into <DIR>/Inkling-2-Mini")
    parser.add_argument("--model-dir", type=Path, help="exact model folder, e.g. an existing download")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the files and report; download and move nothing")
    args = parser.parse_args()
    model = load_model(ROOT / "runtime-manifest.json")
    files = model["files"]
    destination = target_directory(args, model).expanduser().absolute()
    print(f"Model: {model['repo_id']} @ {model['revision']}\nFolder: {destination}", flush=True)

    if args.verify_only:
        if not destination.is_dir():
            raise ValueError(f"No model folder at {destination}")
        print(f"Verifying {len(files)} files ({human(sum(p['size'] for p in files.values()))})...", flush=True)
        results = check(destination, files, list(files))
        report(results)
        bad = [name for name, problem in results.items() if problem]
        if bad:
            raise ValueError(f"{len(bad)} of {len(files)} files are missing or do not match the pinned revision.\n"
                             "Run bash setup.sh and choose 2 to download or repair them.")
        print(f"Model verified: {destination}", flush=True)
        return

    destination.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination / ".ingine-download.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another download into this folder is running; waiting for it...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        present = [name for name in files if (destination / name).exists() or (destination / name).is_symlink()]
        previous = None
        if present:
            print(f"Verifying {len(present)} existing files...", flush=True)
            results = check(destination, files, present)
            report(results)
            for name, problem in results.items():
                if problem:
                    previous = previous or previous_directory(destination)
                    set_aside(destination / name, previous / name)
                    print(f"Moved aside to {previous / name}", flush=True)
        missing = [name for name in files if not (destination / name).exists()]
        if missing:
            needed = sum(files[name]["size"] for name in missing)
            free = shutil.disk_usage(destination).free
            if free < needed + SPACE_MARGIN:
                raise ValueError(f"The remaining files need {human(needed)} of free space in {destination}; "
                                 f"{human(free)} is free. Free some space or choose another --models-dir.")
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise ValueError("huggingface_hub is not installed in this Python. "
                                 "Run bash setup.sh and choose 1.") from error
            print(f"Downloading {len(missing)} file{'s' if len(missing) != 1 else ''} ({human(needed)}); "
                  "an interrupted download resumes when run again...", flush=True)
            # Partial files live in <folder>/.cache/huggingface until complete; files verified above are
            # excluded so they are not fetched again.
            snapshot_download(repo_id=model["repo_id"], revision=model["revision"], local_dir=destination,
                              allow_patterns=missing)
            results = check(destination, files, missing)
            report(results)
            bad = [name for name, problem in results.items() if problem]
            if bad:
                previous = previous or previous_directory(destination)
                for name in bad:
                    if (destination / name).exists():
                        set_aside(destination / name, previous / name)
                raise ValueError(f"{len(bad)} downloaded file(s) do not match the pinned revision and were "
                                 f"moved to {previous}. Retry the download.")
    if previous:
        print(f"Files that did not match are kept in {previous}; delete it once you no longer need them.",
              flush=True)
    print(f"Model ready: {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted; run the download again to resume.")
    except Exception as error:  # network and Hub errors included: one clear line instead of a traceback
        sys.exit(f"Download failed: {error}")
