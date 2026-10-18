#!/bin/sh
# Veritas installer — builds Veritas in an isolated virtualenv and puts the
# `veritas` command on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/Nyvo2010/veritas/main/install.sh | sh
#
# From a checkout, just run ./install.sh (it installs that working tree).
#
# Environment overrides:
#   VERITAS_REPO      GitHub repo (default: Nyvo2010/veritas)
#   VERITAS_REF       branch/tag (default: main)
#   VERITAS_HOME      virtualenv location (default: ~/.local/share/veritas)
#   VERITAS_BIN_DIR   where the `veritas` command is linked (default: ~/.local/bin)
#   VERITAS_FROM_SRC  install from this local directory instead of downloading

set -eu

REPO="${VERITAS_REPO:-Nyvo2010/veritas}"
REF="${VERITAS_REF:-main}"
HOME_DIR="${VERITAS_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/veritas}"
BIN_DIR="${VERITAS_BIN_DIR:-$HOME/.local/bin}"
VENV="$HOME_DIR/venv"

info() { printf '  %s\n' "$1"; }
die() { printf 'error: %s\n' "$1" >&2; exit 1; }

find_python() {
    for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

printf '\nVeritas installer\n\n'

# 1. A suitable Python.
PYTHON="$(find_python || true)"
[ -n "$PYTHON" ] || die "Python 3.10+ is required but was not found on PATH."
info "Python: $($PYTHON --version 2>&1) ($(command -v "$PYTHON"))"

# 2. Where the source comes from: a local checkout, or a downloaded tarball.
SOURCE=""
CLEANUP=""
if [ -n "${VERITAS_FROM_SRC:-}" ]; then
    SOURCE="$VERITAS_FROM_SRC"
elif [ -f "$0" ] && [ -f "$(dirname "$0")/pyproject.toml" ]; then
    SOURCE="$(cd "$(dirname "$0")" && pwd)"
fi

if [ -n "$SOURCE" ]; then
    [ -f "$SOURCE/pyproject.toml" ] || die "No pyproject.toml in $SOURCE"
    info "Source: $SOURCE (local)"
else
    command -v curl >/dev/null 2>&1 || die "curl is required to download Veritas."
    command -v tar >/dev/null 2>&1 || die "tar is required to unpack Veritas."
    TMP="$(mktemp -d "${TMPDIR:-/tmp}/veritas.XXXXXX")"
    CLEANUP="$TMP"
    trap '[ -n "$CLEANUP" ] && rm -rf "$CLEANUP"' EXIT INT TERM
    URL="https://github.com/$REPO/archive/refs/heads/$REF.tar.gz"
    info "Downloading: $URL"
    curl -fsSL "$URL" -o "$TMP/src.tar.gz" || die "Download failed. Check VERITAS_REPO/VERITAS_REF."
    tar -xzf "$TMP/src.tar.gz" -C "$TMP" || die "Could not unpack the archive."
    SOURCE="$(find "$TMP" -maxdepth 1 -type d -name 'veritas-*' | head -n 1)"
    [ -n "$SOURCE" ] || die "Unexpected archive layout."
fi

# 3. Isolated virtualenv.
mkdir -p "$HOME_DIR" "$BIN_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    info "Creating virtualenv: $VENV"
    "$PYTHON" -m venv "$VENV" || die "Could not create a virtualenv (is python3-venv installed?)."
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true

# 4. Install. Try the full TUI first; fall back to the Rich CLI when no
#    prebuilt opentui wheel exists for this Python.
if "$VENV/bin/python" -m pip install --quiet "${SOURCE}[tui]" 2>/dev/null; then
    info "Installed with the full TUI (opentui)."
elif "$VENV/bin/python" -m pip install --quiet "$SOURCE"; then
    info "Installed the Rich CLI (no opentui wheel for this Python)."
else
    die "Installation failed. Re-run with the log visible: $VENV/bin/python -m pip install \"${SOURCE}[tui]\""
fi

# 5. Put `veritas` on PATH.
ln -sf "$VENV/bin/veritas" "$BIN_DIR/veritas"
info "Linked command: $BIN_DIR/veritas"

# Link into /usr/local/bin too when it is writable without sudo.
if [ "$BIN_DIR" != "/usr/local/bin" ] && [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
    ln -sf "$VENV/bin/veritas" /usr/local/bin/veritas
    info "Linked command: /usr/local/bin/veritas"
fi

printf '\nVeritas is installed.\n\n'
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) printf 'Add %s to your PATH, then restart your shell:\n\n  export PATH="%s:$PATH"\n\n' "$BIN_DIR" "$BIN_DIR" ;;
esac
printf 'Run it:\n\n  veritas\n\n'
