#!/usr/bin/env bash
# Install, update, download, start the foreground server, or install the ingine command.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PATH="$ROOT/.tools/bin:$PATH"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash setup.sh [--update]\n\nChoose 1 to install ingine (build tools and a private CUDA 13.4 toolkit, then build the\nC++ server) and download the pinned Inkling-2-Mini checkpoint.\nChoose 2 to download or resume the model download.\nChoose 3 to start the server in the foreground with the manifest profile.\nChoose 4, or run bash setup.sh --update, to update this installer, rebuild the server (only\nwhat changed) and rerun the install steps; the model is downloaded again only when its\npinned revision changed (mismatching files are moved to <model folder>.previous-<time>/,\nnot deleted).\nChoose 5 to install the ingine command (start, stop, status, logs, update) as\n~/.local/bin/ingine.\n\nINGINE_MODEL_DIR reuses an existing download of the model. Serving requires Linux x86_64,\nan RTX 5090 (32 GB) and an NVIDIA driver for CUDA 13 (580 or newer). Allow about 20.2 GB\nfor the model plus about 3 GB for software.\n'
  exit 0
fi
# --update-continue is internal: after --update (or option 4) pulls the installer, the pulled
# setup.sh finishes the update, then shows the menu if --menu follows. Keep accepting it, so
# older copies can hand an update over to newer ones. INGINE_UPDATE_FROM carries the previous commit.
update=""
show_menu=1
case "$*" in
  "") ;;
  --update) update=pull; show_menu=0 ;;
  --update-continue) update="continue"; show_menu=0 ;;
  "--update-continue --menu") update="continue" ;;
  *)
    echo "Unexpected arguments; use bash setup.sh --help for usage." >&2
    exit 2
    ;;
esac
if [[ ! -t 0 ]]; then
  echo "Run bash setup.sh in an interactive terminal (or use --help)." >&2
  exit 2
fi
active_step=""
interrupt_setup() {
  trap '' INT
  if [[ -n "$active_step" ]]; then
    kill -TERM -- "-$active_step" 2>/dev/null || true
    wait "$active_step" 2>/dev/null || true
  fi
  if [[ -n "$update" ]]; then
    printf '\nInterrupted. Run the update again to continue it.\n'
  else
    printf '\nInterrupted. Choose 1 to continue installation or 2 to resume the download.\n'
  fi
  exit 130
}
trap interrupt_setup INT

run_step() {
  # Keep Bash's interrupt trap responsive and stop all download/build workers.
  setsid "$@" &
  active_step=$!
  local status=0
  wait "$active_step" || status=$?
  active_step=""
  return "$status"
}

# A GCC that CUDA 13.4 accepts as its host compiler (up to 16); scripts/install.sh asks nvcc
# itself. An explicit CUDAHOSTCXX (for example a clang++) is left to that check.
has_cuda_host_compiler() {
  local candidate major
  if [[ -n "${CUDAHOSTCXX:-}" ]] && command -v "$CUDAHOSTCXX" >/dev/null; then
    return 0
  fi
  for candidate in c++ g++-16 g++-15 g++-14 g++-13 g++-12; do
    command -v "$candidate" >/dev/null || continue
    [[ "$("$candidate" --version 2>/dev/null | head -n 1)" != *clang* ]] || continue
    major="$("$candidate" -dumpversion 2>/dev/null | cut -d. -f1)"
    if [[ "$major" =~ ^[0-9]+$ ]] && (( major <= 16 )); then
      return 0
    fi
  done
  return 1
}

check_system_tools() {
  local command missing=0
  for command in curl git cc c++ setsid; do
    if ! command -v "$command" >/dev/null; then
      printf 'Missing system tool: %s\n' "$command" >&2
      missing=1
    fi
  done
  if command -v c++ >/dev/null && ! has_cuda_host_compiler; then
    echo "Missing a host compiler for CUDA 13.4 (GCC up to 16); the system compiler is newer." >&2
    missing=1
  fi
  return "$missing"
}

ensure_system_tools() {
  if check_system_tools; then
    return 0
  fi
  local answer
  local package_command=() privilege=()
  if command -v apt-get >/dev/null; then
    package_command=(apt-get install -y build-essential git curl util-linux)
  elif command -v dnf >/dev/null; then
    package_command=(dnf install -y gcc gcc-c++ make git curl util-linux)
  elif command -v pacman >/dev/null; then
    package_command=(pacman -S --needed base-devel git curl util-linux)
    if command -v c++ >/dev/null && ! has_cuda_host_compiler; then
      package_command+=(gcc15)
    fi
  else
    echo "Install curl, git and a C++ compiler (GCC up to 16), then choose 1 again." >&2
    return 1
  fi
  if [[ $EUID -ne 0 ]]; then
    if ! command -v sudo >/dev/null; then
      echo "Ask your administrator to install the missing tools, then choose 1 again." >&2
      return 1
    fi
    privilege=(sudo)
  fi
  printf '\nSystem package command: %s\n' "${privilege[*]} ${package_command[*]}"
  echo "On apt-based systems, package lists will also be refreshed first."
  if ! read -r -p "Install these system tools? [y/N] " answer; then
    return 1
  fi
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "No system packages changed."; return 1 ;;
  esac
  if [[ "${package_command[0]}" == apt-get ]]; then
    "${privilege[@]}" apt-get update || return 1
  fi
  "${privilege[@]}" "${package_command[@]}" || return 1
  if ! check_system_tools; then
    echo "If only the CUDA host compiler is missing, install GCC 12-16 (for example g++-14) or set CUDAHOSTCXX." >&2
    return 1
  fi
}

download_model() {
  local command
  for command in curl git setsid; do
    if ! command -v "$command" >/dev/null; then
      printf 'Missing %s. Install it, or choose 1 for system prerequisite setup.\n' "$command" >&2
      return 1
    fi
  done
  run_step bash "$ROOT/scripts/install.sh" --download-only
}

next_steps() {
  printf 'Start the server with option 3 (foreground), or with ingine start in the background\n'
  printf '(option 5 installs the ingine command). API: http://127.0.0.1:8003/v1\n'
}

install_everything() {
  ensure_system_tools || return 1
  run_step bash "$ROOT/scripts/install.sh" || return 1
  printf '\nInstallation complete. '
  next_steps
}

install_command() {
  local bin="$HOME/.local/bin" link="$HOME/.local/bin/ingine" blink="$HOME/.local/bin/blackink" target="$ROOT/scripts/ingine"
  mkdir -p -- "$bin" || return 1
  if [[ -L "$link" && "$(readlink -- "$link")" == "$target" ]]; then
    echo "The ingine command is already installed: $link -> $target"
  elif [[ -e "$link" || -L "$link" ]]; then
    echo "$link already exists and is not this installer's command. Move it aside, then choose 5 again; it was not changed." >&2
    return 1
  else
    ln -s -- "$target" "$link" || return 1
    echo "Installed the ingine command: $link -> $target"
  fi
  ln -sf -- "$target" "$blink" || true
  echo "Installed the blackink command: $blink -> $target"
  case ":$PATH:" in
    *":$bin:"*) ;;
    *)
      echo "Warning: $bin is not on your PATH. Add this line to ~/.bashrc (or your shell's profile), then open a new terminal:"
      # shellcheck disable=SC2016  # printed literally for the user's profile
      printf '  export PATH="$HOME/.local/bin:$PATH"\n'
      ;;
  esac
  printf 'Use: ingine start | stop | status | logs | serve | update (ingine help for details)\n'
}

# Fast-forwards this clone to the published installer, whose manifest pins the current model.
# Local edits that the update does not touch are kept.
update_installer() {
  local top before after
  if ! command -v git >/dev/null; then
    echo "Missing git. Install it, then run the update again." >&2
    return 1
  fi
  top="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" || top=""
  if [[ -z "$top" || ! "$top" -ef "$ROOT" ]]; then
    echo "This folder is not a git clone of the installer, so it cannot update itself." >&2
    echo "Clone https://github.com/zaptize-dev/blackink and install from it." >&2
    return 1
  fi
  before="$(git -C "$ROOT" rev-parse HEAD)" || return 1
  printf '\nFetching the latest installer...\n'
  git -C "$ROOT" fetch --quiet origin main || return 1
  if ! git -C "$ROOT" merge --quiet --ff-only FETCH_HEAD; then
    echo "Local commits or edits in this folder conflict with the update; nothing was changed." >&2
    echo "Review them with git status, save or undo them, then run the update again." >&2
    return 1
  fi
  after="$(git -C "$ROOT" rev-parse HEAD)" || return 1
  if [[ "$after" == "$before" ]]; then
    echo "The installer is already up to date."
  else
    printf 'Installer updated from %s to %s:\n' "${before:0:7}" "${after:0:7}"
    git -C "$ROOT" log -n 10 --format='  %s' "$before..$after" || true
  fi
  export INGINE_UPDATE_FROM="$before"
}

# Reruns the idempotent install steps at the new pins; the model only when its revision changed.
finish_update() {
  ensure_system_tools || return 1
  run_step bash "$ROOT/scripts/install.sh" --update-from "${INGINE_UPDATE_FROM:-}" || return 1
  printf '\nUpdate complete. Restart a running server so it uses the update: ingine stop && ingine start\n'
  if [[ "$show_menu" == 1 ]]; then
    echo "(or, for a server started with option 3, stop it with Ctrl-C and choose 3 again)."
  fi
}

if [[ "$update" == pull ]]; then
  if ! update_installer; then
    printf '\nUpdate did not finish. Fix the error above, then run it again.\n' >&2
    exit 1
  fi
  exec bash "$ROOT/setup.sh" --update-continue
fi
if [[ "$update" == continue ]]; then
  if ! finish_update; then
    printf '\nUpdate did not finish. Fix the error above, then run it again.\n' >&2
    [[ "$show_menu" == 1 ]] || exit 1
  fi
  [[ "$show_menu" == 1 ]] || exit 0
  update=""
  unset INGINE_UPDATE_FROM
fi

while true; do
  printf '\nBlackInk — Inkling-2-Mini abliterated (16 GB Blackwell Edition)\n'
  printf 'RTX 5060 Ti 16GB+, 131,072-token context (Q4), Full-NVFP4 experts, DSpark speculative decoding + Web UI\n\n'
  printf '  1) Install everything (software + C++ engine build + model)\n'
  printf '  2) Download / resume model\n'
  printf '  3) Start the server (Ctrl-C to stop)\n'
  printf '  4) Update to the latest version\n'
  printf '  5) Install the blackink / ingine commands (~/.local/bin/blackink)\n'
  printf '  0) Exit\n\n'
  printf 'Space: ~20.2 GB model files, plus ~3 GB software.\n'
  if ! read -r -p "Choose an option: " choice; then
    printf '\n'
    exit 0
  fi
  case "$choice" in
    1)
      if ! install_everything; then
        printf '\nSetup did not finish. Fix the error above, then choose 1 again.\n' >&2
      fi
      ;;
    2)
      if ! download_model; then
        printf '\nDownload did not finish. Fix the error above, then retry.\n' >&2
      fi
      ;;
    3)
      if [[ ! -x "$ROOT/native/build-install/apps/serve/ingine-serve" || ! -r "$ROOT/native/build-install/profile.env" ]]; then
        echo "Choose 1 to install the engine and download the model first." >&2
      else
        exec bash "$ROOT/scripts/ingine" serve
      fi
      ;;
    4)
      update=pull
      if update_installer; then
        exec bash "$ROOT/setup.sh" --update-continue --menu
      fi
      update=""
      printf '\nUpdate did not finish. Fix the error above, then choose 4 again.\n' >&2
      ;;
    5)
      if ! install_command; then
        printf '\nThe ingine command was not installed. Fix the error above, then choose 5 again.\n' >&2
      fi
      ;;
    0|q|Q) exit 0 ;;
    *) echo "Choose 1, 2, 3, 4, 5, or 0." ;;
  esac
done
