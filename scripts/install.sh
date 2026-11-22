#!/usr/bin/env bash
# Install the build tools and the private CUDA toolkit, build ingine's C++ server, then download the model.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/install.sh [--download-only | --update-from COMMIT]

Without options: check the system, install uv (into .tools/ when missing), the installer's
Python environment in .tools/venv/ (CMake, Ninja and huggingface_hub), the CUDA 13.4 toolkit
with cuBLAS in .cuda-toolkit/, build ingine's C++ server into native/build-install/, then
download and verify the pinned model (Inkling-2-Mini weights and the ablation, about 20.2 GB)
into models/Inkling-2-Mini. The server runs without Python or PyTorch.

--download-only    only what downloading needs (uv, .tools/venv with huggingface_hub), then
                   download or resume the model; menu option 2.
--update-from REV  used by bash setup.sh --update: reinstall the software steps, rebuild what
                   changed in the server, and download the model only when the manifest's model
                   revision differs from the one at installer commit REV (or the files are
                   incomplete).

INGINE_MODEL_DIR (environment or ~/.config/ingine/env) selects an existing download; a complete
one is reused as is. When it comes from the environment it is saved to ~/.config/ingine/env so
the ingine command serves from it.
CUDAHOSTCXX selects the host compiler nvcc uses; by default CXX when set, else the first of c++,
g++, g++-16, g++-15, g++-14, g++-13 and g++-12 that the pinned nvcc accepts. CXX (default: that
host compiler) compiles the C++ code, CC (default: cc) the C code.
Requires Linux x86_64, an NVIDIA driver for CUDA 13 (580 or newer), git, curl and a C++20
compiler. The installer does not install or change GPU drivers.
HELP
  exit 0
fi

MODE=full
UPDATE_FROM=""
if [[ $# == 1 && "$1" == --download-only ]]; then
  MODE=download
elif [[ $# == 2 && "$1" == --update-from ]]; then
  MODE=update
  UPDATE_FROM="$2"
elif [[ $# != 0 ]]; then
  fail 'Unexpected arguments; use --help for usage.'
fi

# The CUDA packages and build tools are pinned in runtime-manifest.json ("runtime").
TOOLS="$ROOT/.tools/venv"
CUDA_HOME="$ROOT/.cuda-toolkit/nvidia/cu13"
BUILD="$ROOT/native/build-install"  # the "install" preset in native/CMakePresets.json
SERVER="$BUILD/apps/serve/ingine-serve"
PROFILE="$BUILD/profile.env"
MIN_DRIVER=580
MIN_GPU_MIB=15360
CONFIG="$HOME/.config/ingine/env"

confirm() {
  local answer
  printf 'Warning: %s\n' "$1" >&2
  [[ -t 0 ]] || fail 'Run the installer in a terminal to confirm this warning (bash setup.sh).'
  read -r -p 'Continue anyway? [y/N] ' answer || answer=""
  case "$answer" in
    y|Y|yes|YES) ;;
    *) fail 'Stopped at your request.' ;;
  esac
}

# Nearest existing folder of a path that may not exist yet.
existing_parent() {
  local path="$1"
  while [[ ! -e "$path" ]]; do
    path="$(dirname -- "$path")"
  done
  printf '%s\n' "$path"
}

# ---- checks before any change ----------------------------------------------------------------------

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'ingine requires Linux x86_64 with an NVIDIA RTX 5090.'
for command in git curl; do
  command -v "$command" >/dev/null || fail "Missing $command. Install it, or run bash setup.sh and choose 1."
done
[[ -r "$ROOT/runtime-manifest.json" ]] || fail "Missing $ROOT/runtime-manifest.json; this folder is not a complete installer clone."

if [[ "$MODE" != download ]]; then
  CXX_COMMAND="${CXX:-c++}"
  command -v "$CXX_COMMAND" >/dev/null || fail "Missing a C++20 compiler ($CXX_COMMAND). Run bash setup.sh and choose 1."
  CC_PATH="$(command -v "${CC:-cc}")" || fail "Missing a C compiler (${CC:-cc}). Run bash setup.sh and choose 1."
  command -v nvidia-smi >/dev/null || fail 'Install your NVIDIA driver first. This installer does not install or change drivers.'
  gpus="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits)" \
    || fail 'The NVIDIA driver is not working (nvidia-smi failed). Fix it before continuing.'
  printf 'Detected GPU(s):\n%s\n' "$gpus"
  driver="$(head -n 1 <<< "$gpus" | awk -F', ' '{print $5}')"
  driver_major="${driver%%.*}"
  [[ "$driver_major" =~ ^[0-9]+$ ]] || fail "Cannot read the NVIDIA driver version from nvidia-smi ($driver)."
  (( driver_major >= MIN_DRIVER )) || fail "NVIDIA driver $driver is too old: ingine is built with CUDA 13, which needs driver $MIN_DRIVER or newer. Update the driver, then rerun."
  if ! awk -F', ' -v min="$MIN_GPU_MIB" '$3 == "12.0" && $4 + 0 >= min {found = 1} END {exit !found}' <<< "$gpus"; then
    confirm 'BlackInk targets a 16+ GB NVIDIA Blackwell GPU (RTX 5060 Ti 16GB / 5070 Ti / 5080 / 5090, compute capability 12.0, sm_120a); its kernels do not run on non-Blackwell GPUs.'
  fi
fi

# INGINE_MODEL_DIR: environment, then ~/.config/ingine/env, then models/<directory>; relative to this
# folder, as in scripts/ingine.
env_model_dir="${INGINE_MODEL_DIR:-}"
config_model_dir=""
if [[ -r "$CONFIG" ]]; then
  # shellcheck source=/dev/null
  config_model_dir="$(unset INGINE_MODEL_DIR; set +eu; source "$CONFIG" >/dev/null 2>&1; printf '%s' "${INGINE_MODEL_DIR:-}")" || config_model_dir=""
fi
model_setting="${env_model_dir:-$config_model_dir}"

# ---- uv and the installer's Python environment -----------------------------------------------------
# Python is used only here: to read the manifest, run CMake and Ninja, and download the model.

if [[ -x "$ROOT/.tools/bin/uv" ]]; then
  UV="$ROOT/.tools/bin/uv"
elif command -v uv >/dev/null; then
  UV="$(command -v uv)"
else
  echo "Installing uv from https://astral.sh/uv/install.sh into .tools/bin (no shell profile changes)."
  installer="$(mktemp)"
  if ! curl --fail --show-error --location --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh --output "$installer"; then
    rm -f -- "$installer"
    fail 'Could not download the uv installer. Check your connection and rerun.'
  fi
  status=0
  UV_UNMANAGED_INSTALL="$ROOT/.tools/bin" sh "$installer" || status=$?
  rm -f -- "$installer"
  [[ $status == 0 && -x "$ROOT/.tools/bin/uv" ]] || fail 'Installing uv failed; see the error above.'
  UV="$ROOT/.tools/bin/uv"
fi

[[ ! -L "$TOOLS" ]] || fail "Refusing to modify a symlinked installation directory: $TOOLS. Use a repository-local installation."
if [[ ! -x "$TOOLS/bin/python" ]]; then
  [[ ! -e "$TOOLS" ]] || fail 'An incomplete .tools/venv already exists. Move it aside before installing; no files were removed.'
  "$UV" venv --quiet --python 3.12 "$TOOLS"
fi
PYTHON="$TOOLS/bin/python"

mapfile -t pin < <("$PYTHON" - "$ROOT/runtime-manifest.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
model, serving, runtime = manifest["model"], manifest["serving"], manifest["runtime"]
if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
    sys.exit("The model revision must be a full commit in runtime-manifest.json")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", model["directory"]):
    sys.exit("Invalid model directory in runtime-manifest.json")
for spec in runtime["cuda_packages"] + runtime["build_tools"]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", spec):
        sys.exit(f"Invalid pin in runtime-manifest.json: {spec}")
size_gb = sum(meta["size"] for meta in model["files"].values()) / 1e9
for value in (model["revision"], model["directory"], f"{size_gb:.1f}", serving["host"], serving["port"], serving["model_id"],
              runtime["cuda"], " ".join(runtime["cuda_packages"]), " ".join(runtime["build_tools"])):
    print(value)
PY
)
[[ ${#pin[@]} == 9 ]] || fail 'Cannot read runtime-manifest.json.'
REVISION="${pin[0]}"
MODEL_DIR="${model_setting:-models/${pin[1]}}"
[[ "$MODEL_DIR" == /* ]] || MODEL_DIR="$ROOT/$MODEL_DIR"
CUDA_VERSION="${pin[6]}"
read -r -a CUDA_PACKAGES <<< "${pin[7]}"
read -r -a BUILD_TOOLS <<< "${pin[8]}"

# Bytes of the pinned files that are missing or have the wrong size in MODEL_DIR.
model_missing_bytes() {
  "$PYTHON" - "$ROOT/runtime-manifest.json" "$MODEL_DIR" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    files = json.load(stream)["model"]["files"]
missing = 0
for name, meta in files.items():
    try:
        size = os.path.getsize(os.path.join(sys.argv[2], name))
    except OSError:
        size = 0
    if size != meta["size"]:
        missing += meta["size"] - size if size < meta["size"] else meta["size"]
print(missing)
PY
}

# The model step runs download_model.py, which resumes, verifies against the pinned revision and moves
# mismatching files aside (never deletes). A complete INGINE_MODEL_DIR, or an unchanged model during an
# update, is kept without rehashing.
missing_bytes="$(model_missing_bytes)"
download=1
if [[ "$MODE" != download && "$missing_bytes" == 0 ]]; then
  if [[ "$MODE" == update ]]; then
    previous_revision=""
    if [[ -n "$UPDATE_FROM" ]]; then
      previous_revision="$(git -C "$ROOT" show "$UPDATE_FROM:runtime-manifest.json" 2>/dev/null \
        | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["model"]["revision"])' 2>/dev/null)" || previous_revision=""
    fi
    [[ "$previous_revision" != "$REVISION" ]] || download=0
  elif [[ -n "$model_setting" ]]; then
    download=0
  fi
fi

# ---- free space --------------------------------------------------------------------------------------

declare -A need_kib=() where=()
add_need() {  # path, KiB
  local parent device
  parent="$(existing_parent "$1")"
  device="$(stat -c %d -- "$parent")"
  need_kib[$device]=$(( ${need_kib[$device]:-0} + $2 ))
  where[$device]="$parent"
}
if [[ "$MODE" != download ]]; then
  [[ -f "$CUDA_HOME/lib/libcublasLt.so.13" && -x "$TOOLS/bin/cmake" ]] || add_need "$ROOT" $(( 3 * 1048576 ))
  [[ -x "$SERVER" ]] || add_need "$ROOT" 1048576
fi
if [[ "$download" == 1 && "$missing_bytes" != 0 ]]; then
  add_need "$MODEL_DIR" $(( missing_bytes / 1024 + 1048576 ))
fi
for device in "${!need_kib[@]}"; do
  free_kib="$(df -Pk -- "${where[$device]}" | awk 'NR == 2 {print $4}')"
  if (( free_kib < need_kib[$device] )); then
    confirm "$(printf '%s has %d GiB free; this step needs about %d GiB there.' \
      "${where[$device]}" $(( free_kib / 1048576 )) $(( (need_kib[$device] + 1048575) / 1048576 )))"
  fi
done

# ---- software ----------------------------------------------------------------------------------------

if [[ "$MODE" == download ]]; then
  if ! "$PYTHON" -c 'import huggingface_hub' 2>/dev/null; then
    "$UV" pip install --python "$PYTHON" 'huggingface_hub[hf_xet]'
  fi
else
  printf '\nInstalling the build tools (%s) and huggingface_hub into .tools/venv/...\n' "${BUILD_TOOLS[*]}"
  "$UV" pip install --python "$PYTHON" 'huggingface_hub[hf_xet]' "${BUILD_TOOLS[@]}"

  printf '\nInstalling the CUDA %s toolkit and cuBLAS into .cuda-toolkit/ (about 2 GB on first install)...\n' "$CUDA_VERSION"
  [[ ! -L "$ROOT/.cuda-toolkit" ]] || fail 'Refusing to modify a symlinked .cuda-toolkit directory.'
  "$UV" pip install --python "$PYTHON" --target "$ROOT/.cuda-toolkit" --no-deps "${CUDA_PACKAGES[@]}"
  # The previous (Python) engine's build settings; nothing reads them any more.
  rm -f -- "$ROOT/.cuda-toolkit/ingine.env"

  # NVIDIA's wheels omit the layout aliases CMake's CUDA detection and linking need. Keep existing
  # correct aliases, and refuse to overwrite unrelated files or symlinks.
  ensure_link() {
    local target="$1" link="$2"
    if [[ -e "$link" || -L "$link" ]]; then
      [[ "$link" -ef "$(dirname -- "$link")/$target" ]] || fail "Unexpected CUDA layout at $link. Move it aside and rerun; it was not overwritten."
    else
      ln -s "$target" "$link"
    fi
  }
  ensure_link lib "$CUDA_HOME/lib64"
  ensure_link libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
  ensure_link libcublasLt.so.13 "$CUDA_HOME/lib/libcublasLt.so"
  for file in bin/nvcc include/cuda_runtime.h include/cuda_fp8.h include/cublasLt.h include/nvtx3/nvToolsExt.h \
    include/crt/host_config.h include/cccl/cuda/std/version nvvm/libdevice/libdevice.10.bc lib/libcudart.so.13 \
    lib/libcublasLt.so.13; do
    [[ -f "$CUDA_HOME/$file" ]] || fail "Incomplete CUDA toolkit: $file is missing. Rerun the installation."
  done
  [[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"V$CUDA_VERSION"* ]] || fail "The private nvcc is not CUDA $CUDA_VERSION."

  # nvcc accepts only some host compilers (CUDA 13.4: GCC up to 16). Ask the pinned nvcc itself.
  probe="$(mktemp -d)"
  printf '#include <concepts>\n#include <span>\n__global__ void k() {}\nint main() { k<<<1, 1>>>(); return 0; }\n' > "$probe/probe.cu"
  if [[ -n "${CUDAHOSTCXX:-}" ]]; then
    candidates=("$CUDAHOSTCXX")
  elif [[ -n "${CXX:-}" ]]; then
    candidates=("$CXX")
  else
    candidates=(c++ g++ g++-16 g++-15 g++-14 g++-13 g++-12)
  fi
  NVCC_HOST=""
  for candidate in "${candidates[@]}"; do
    path="$(command -v "$candidate")" || continue
    if "$CUDA_HOME/bin/nvcc" -ccbin "$path" -std=c++20 -gencode=arch=compute_120a,code=sm_120a \
      -c "$probe/probe.cu" -o "$probe/probe.o" >/dev/null 2>&1; then
      NVCC_HOST="$path"
      break
    fi
  done
  rm -rf -- "$probe"
  if [[ -z "$NVCC_HOST" ]]; then
    [[ -z "${CUDAHOSTCXX:-}" ]] || fail "CUDA $CUDA_VERSION cannot compile C++20 with CUDAHOSTCXX=$CUDAHOSTCXX."
    [[ -z "${CXX:-}" ]] || fail "CUDA $CUDA_VERSION cannot compile C++20 with CXX=$CXX. Set CUDAHOSTCXX to a GCC up to 16, or unset CXX, then rerun."
    fail "No C++20 host compiler that CUDA $CUDA_VERSION accepts (GCC up to 16). Install one (for example g++-14 on Debian/Ubuntu) or set CUDAHOSTCXX, then rerun."
  fi
  CXX_PATH="$(command -v "${CXX:-$NVCC_HOST}")"
  printf 'CUDA host compiler: %s\n' "$NVCC_HOST"

  # One compile job per 2 GiB of available memory (the largest kernel files need about 1 GiB in nvcc),
  # at most one per CPU.
  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  jobs=$(( available_kib / (2 * 1048576) ))
  (( jobs <= $(nproc) )) || jobs=$(nproc)
  (( jobs >= 1 )) || jobs=1

  # The "install" preset puts the build in native/build-install/ with the private toolkit, and links the
  # binaries with a relative RPATH into .cuda-toolkit/, so the server needs no system CUDA, LD_LIBRARY_PATH or
  # Python. Ninja rebuilds only what changed.
  [[ ! -L "$BUILD" ]] || fail "Refusing to modify a symlinked build directory: $BUILD."
  printf '\nBuilding the C++ engine (ingine-serve, %d parallel jobs) into native/build-install/; the first build takes a few minutes...\n' "$jobs"
  if ! (cd "$ROOT/native" && "$TOOLS/bin/cmake" --preset install --log-level=WARNING \
    -DCMAKE_MAKE_PROGRAM="$TOOLS/bin/ninja" -DCMAKE_C_COMPILER="$CC_PATH" -DCMAKE_CXX_COMPILER="$CXX_PATH" \
    -DCMAKE_CUDA_HOST_COMPILER="$NVCC_HOST" >/dev/null); then
    fail 'Configuring the C++ build failed; see the error above. If native/build-install was configured in another folder or with other compilers, move it aside and rerun; it was not deleted.'
  fi
  (cd "$ROOT/native" && "$TOOLS/bin/cmake" --build --preset install --target ingine-serve --parallel "$jobs") \
    || fail 'Building the C++ engine failed; see the error above. Rerun to retry; finished parts are kept.'
  "$SERVER" --help >/dev/null || fail "The built server does not run ($SERVER); see the error above."

  # scripts/ingine reads the serving profile from here, so serving needs no JSON parser.
  "$PYTHON" - "$ROOT/runtime-manifest.json" > "$PROFILE.tmp" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
serving, model = manifest["serving"], manifest["model"]
flags = ["--" + key.replace("_", "-") for key in ("dense_fp8", "full_nvfp4", "kv_fp8", "kv_q4", "spec") if serving.get(key)]
if serving.get("spec"):
    flags += ["--spec-threshold", str(serving["spec_threshold"])]
# Files the server reads; the ablation file is checked on its own because INGINE_ABLATION may replace it.
skip = {"README.md", "LICENSE", "NOTICE", "SHA256SUMS", ".gitattributes", manifest["ablation"]}
required = [name for name in dict.fromkeys(["config.json", *model["files"]]) if name not in skip]
print("# Written by scripts/install.sh from runtime-manifest.json; sourced by scripts/ingine.")
for name, value in (("host", serving["host"]), ("port", serving["port"]), ("max_ctx", serving["max_ctx"]),
                    ("model_id", serving["model_id"]), ("ablation", manifest["ablation"]),
                    ("directory", model["directory"])):
    print(f"profile_{name}={shlex.quote(str(value))}")
print("profile_flags=(" + " ".join(map(shlex.quote, flags)) + ")")
print("profile_required=(" + " ".join(map(shlex.quote, required)) + ")")
PY
  mv -f -- "$PROFILE.tmp" "$PROFILE"
  echo 'The C++ engine is built.'
fi

# ---- model -------------------------------------------------------------------------------------------

if [[ "$download" == 1 ]]; then
  printf '\nDownloading and verifying the model at revision %s (about %s GB) into %s...\n' "${REVISION:0:12}" "${pin[2]}" "$MODEL_DIR"
  "$PYTHON" "$ROOT/scripts/download_model.py" --model-dir "$MODEL_DIR" \
    || fail 'The model download did not finish. Run it again to resume (bash setup.sh, option 2).'
elif [[ "$MODE" == update ]]; then
  printf '\nThe model is unchanged (revision %s) and complete in %s.\n' "${REVISION:0:12}" "$MODEL_DIR"
else
  printf '\nUsing the complete existing download in %s (file sizes match revision %s).\n' "$MODEL_DIR" "${REVISION:0:12}"
  printf 'Check its hashes with: %q %q --model-dir %q --verify-only\n' "$PYTHON" "$ROOT/scripts/download_model.py" "$MODEL_DIR"
fi

# A model folder given only in the environment is remembered for the ingine command.
if [[ -n "$env_model_dir" && "$env_model_dir" != "$config_model_dir" ]]; then
  if [[ -n "$config_model_dir" ]]; then
    printf 'Note: %s sets INGINE_MODEL_DIR=%s; the ingine command serves from that folder unless INGINE_MODEL_DIR is set.\n' "$CONFIG" "$config_model_dir"
  elif [[ ! -e "$CONFIG" ]] || ! grep -q 'INGINE_MODEL_DIR' "$CONFIG"; then
    mkdir -p -- "$(dirname -- "$CONFIG")"
    printf 'INGINE_MODEL_DIR=%q\n' "$MODEL_DIR" >> "$CONFIG"
    printf 'Saved INGINE_MODEL_DIR=%s to %s for the ingine command.\n' "$MODEL_DIR" "$CONFIG"
  fi
fi

if [[ "$MODE" == download ]]; then
  printf '\nThe model is ready. Choose 1 in bash setup.sh to install the engine if you have not yet.\n'
else
  # The Python engine's environment and leftovers (compiled kernels, caches) from earlier versions are not
  # deleted automatically; engine/ is no longer part of the repository.
  previous=()
  for path in .venv engine; do
    [[ ! -d "$ROOT/$path" ]] || previous+=("$path")
  done
  if (( ${#previous[@]} )); then
    printf '\nThe previous Python engine is no longer used; delete its files to free %s: cd %q && rm -rf %s\n' \
      "$(cd "$ROOT" && du -sch -- "${previous[@]}" 2>/dev/null | awk 'END {print $1}')" "$ROOT" "${previous[*]}"
  fi
  printf '\nReady. Start the server with bash setup.sh, option 3 (foreground), or with ingine start\n'
  printf '(background; option 5 installs the ingine command). API: http://%s:%s/v1, model %s\n' "${pin[3]}" "${pin[4]}" "${pin[5]}"
fi
