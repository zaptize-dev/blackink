#!/usr/bin/env python3
"""Pin the Hugging Face model repo in runtime-manifest.json.

Sets model.revision to the repo's commit (the current head of main, or --revision) and model.files to
{path: {size, sha256}} for every file the installer downloads: all files at that commit except the
provenance folder abliteration/ and .gitattributes. LFS files take size and SHA-256 from the Hub's LFS
metadata; small non-LFS files are downloaded at that commit and hashed. Other manifest keys are kept.
A token, if the repo needs one, comes from HF_TOKEN.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runtime-manifest.json"


def excluded(name):
    return name == ".gitattributes" or name.startswith("abliteration/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--revision", default="main", help="branch, tag or commit to pin (default: main)")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = manifest["model"]
    token = os.environ.get("HF_TOKEN") or None
    info = HfApi(token=token).model_info(model["repo_id"], revision=args.revision, files_metadata=True)
    commit = info.sha
    print(f"{model['repo_id']} @ {args.revision} = {commit}")
    files = {}
    with tempfile.TemporaryDirectory() as cache:
        for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
            name = sibling.rfilename
            if excluded(name):
                continue
            if sibling.lfs is not None:
                files[name] = {"size": sibling.lfs.size, "sha256": sibling.lfs.sha256}
            else:
                path = Path(hf_hub_download(model["repo_id"], name, revision=commit, cache_dir=cache, token=token))
                data = path.read_bytes()
                if sibling.size is not None and len(data) != sibling.size:
                    raise SystemExit(f"{name}: downloaded {len(data)} bytes, the Hub lists {sibling.size}")
                files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            print(f"  {name}: {files[name]['size']} bytes, {files[name]['sha256']}")
    ablation = manifest.get("ablation")
    if ablation and ablation not in files:
        raise SystemExit(f"The repo has no {ablation} (manifest key 'ablation'); nothing written.")
    model["revision"], model["files"] = commit, files
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned {len(files)} files ({sum(f['size'] for f in files.values()) / 1e9:.1f} GB) in {MANIFEST}")


if __name__ == "__main__":
    main()
