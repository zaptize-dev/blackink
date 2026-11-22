#!/usr/bin/env bash
# One-line install:
#   curl -fsSL https://raw.githubusercontent.com/zaptize-dev/blackink/main/get.sh | bash
# Clones this installer to ~/blackink (or $INGINE_DIR) and opens its menu: choose 1
# to install, then 3 to start the server. Running it again reopens the menu of the existing clone.
# Everything happens inside main, which runs only once this whole file has been downloaded.
set -euo pipefail

main() {
  local repo=https://github.com/zaptize-dev/blackink.git
  local dir="${INGINE_DIR:-$HOME/blackink}"

  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This installer needs Linux x86_64 with an NVIDIA Blackwell GPU (RTX 5060 Ti 16GB+)." >&2
    exit 1
  fi
  if ! command -v git >/dev/null; then
    echo "git is required. Install it (sudo apt install git, sudo dnf install git or sudo pacman -S git), then run this again." >&2
    exit 1
  fi

  if [[ -e "$dir" ]]; then
    if [[ "$(git -C "$dir" remote get-url origin 2>/dev/null)" != *zaptize-dev/blackink* ]]; then
      echo "$dir exists and is not this installer. Choose another folder: curl ... | INGINE_DIR=/path/to/folder bash" >&2
      exit 1
    fi
    echo "Using the installer already in $dir (menu option 4 updates it)."
  else
    echo "Downloading the installer to $dir ..."
    git clone --quiet "$repo" "$dir"
  fi

  # Piped into bash, standard input is this script, not the keyboard: give the menu the terminal.
  if ! { true </dev/tty; } 2>/dev/null; then
    printf 'No terminal for the menu. Run it yourself:\n  cd %q && bash setup.sh\n' "$dir"
    exit 0
  fi
  cd "$dir"
  exec bash setup.sh </dev/tty
}

main "$@"
