#!/usr/bin/env bash
# Apply local patches that the viewer relies on.
#
# 1) tribev2.patch — adds Japanese support and the `--task translate` route to
#    facebookresearch/tribev2 (must be cloned and editable-installed already).
# 2) neuralset.patch — adds Japanese to neuralset's spaCy language map (the
#    package that is pulled in transitively as a tribev2 dependency).
#
# Re-running this script is a no-op once the patches are applied.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${VIEWER_ROOT:-$(cd "$HERE/.." && pwd)}"

apply() {
  local target="$1" patch="$2"
  if patch -p1 --dry-run -d "$target" -f --silent < "$patch" >/dev/null 2>&1; then
    echo "applying  $patch  ->  $target"
    patch -p1 -d "$target" < "$patch"
  else
    if patch -p1 -R --dry-run -d "$target" -f --silent < "$patch" >/dev/null 2>&1; then
      echo "skipping  $patch  (already applied to $target)"
    else
      echo "ERROR     $patch  cannot be applied or reversed cleanly against $target" >&2
      exit 1
    fi
  fi
}

# tribev2 source. Resolve from VIEWER_TRIBE_SRC, fall back to ./tribev2-src.
TRIBE_SRC="${VIEWER_TRIBE_SRC:-$ROOT/tribev2-src}"
if [[ ! -d "$TRIBE_SRC/tribev2" ]]; then
  echo "ERROR: tribev2 sources not found at $TRIBE_SRC. Set VIEWER_TRIBE_SRC=path." >&2
  exit 1
fi
apply "$TRIBE_SRC" "$HERE/tribev2.patch"

# neuralset is inside the active venv site-packages. Locate it via Python.
SITE_PACKAGES="$(python -c 'import neuralset, os, pathlib; print(pathlib.Path(neuralset.__file__).parent.parent)')"
if [[ -z "$SITE_PACKAGES" ]]; then
  echo "ERROR: cannot locate neuralset; activate the viewer venv first." >&2
  exit 1
fi
apply "$SITE_PACKAGES" "$HERE/neuralset.patch"

echo "done."
