#!/bin/bash
# Refresh the released code from the project working tree.
#
#   usage: ./sync.sh [PROJECT_DIR]        default: ../.. (the release lives in <project>/release/DLO-Anything)
#
# Copies a fixed list of files (dlo/ package, scripts/) and re-applies the path adaptations that make the
# scripts work from scripts/ inside this repository. Running it twice gives the same result.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "${1:-$HERE/../..}" && pwd)"
[ -f "$SRC/dlo/onestep.py" ] || { echo "no project tree at $SRC (dlo/onestep.py not found)" >&2; exit 1; }
[ "$SRC" != "$HERE" ] || { echo "source and release directory are the same" >&2; exit 1; }

DLO="__init__ config cable_env two_cable_env u_env generate_data generate_u generate_u_ood dataset model train
     contact_model soft_contact train_chain onestep make_videos make_videos_two make_videos_overlay
     make_videos_force make_videos_soft
     couple contact_data force_model"                       # last line: modules imported by the ones above
SCRIPTS="pipeline_onestep.sh gen.sh wrap.py make_motion_sets.py make_grid.py make_u3.py eval_h.py eval_motion.py diag_contact.py diag_memo.py diag_noise.py"

# absolute paths of the working tree become relative paths
strip_abs() { sed -i -E 's#/home/[A-Za-z0-9_.-]+/Python/DLOanything_fs/?##g' "$1"; }

# wording of one docstring in dlo/config.py (comment only): "generat-ed with" -> "created with"
reword() { sed -i -E 's#they were generat[e]d with#they were created with#g' "$1"; }

mkdir -p "$HERE/dlo" "$HERE/scripts"
for m in $DLO; do
  cp "$SRC/dlo/$m.py" "$HERE/dlo/$m.py"; strip_abs "$HERE/dlo/$m.py"; reword "$HERE/dlo/$m.py"
done
for s in $SCRIPTS; do
  cp "$SRC/runs/u3/$s" "$HERE/scripts/$s"
done

# scripts/*.sh: run from the repository root (one level above scripts/), call the helpers in scripts/
sed -i -E 's#^cd /home/[^;]*#cd "$(dirname "$0")/.."#' "$HERE/scripts/gen.sh"
sed -i 's#runs/u3/make_u3\.py#scripts/make_u3.py#g' "$HERE/scripts/gen.sh"
sed -i -e 's#cd "$(dirname "$0")/\.\./\.\."#cd "$(dirname "$0")/.."#' \
       -e 's#runs/u3/pipeline_onestep\.sh#scripts/pipeline_onestep.sh#g' \
       -e 's#runs/u3/wrap\.py#scripts/wrap.py#g' "$HERE/scripts/pipeline_onestep.sh"
for s in $SCRIPTS; do strip_abs "$HERE/scripts/$s"; done
chmod +x "$HERE/scripts/"*.sh

# checks: no absolute home paths left, every imported dlo module is part of the release
if grep -rn "/home/" "$HERE/dlo" "$HERE/scripts" --include=*.py --include=*.sh; then
  echo "WARNING: absolute paths left in the files listed above" >&2
fi
missing=$(cd "$HERE" && grep -hoE '^\s*from (\.|dlo\.)[a-z_0-9]+ import|^\s*from (\.|dlo) import [a-z_0-9, ]+' dlo/*.py scripts/*.py \
  | sed -E 's/^\s*from (\.|dlo\.)([a-z_0-9]+) import/\2/; s/^\s*from (\.|dlo) import //; s/,/ /g; s/ as [a-z_0-9]+//g' | tr ' ' '\n' | sort -u \
  | while read -r m; do [ -z "$m" ] || [ -f "dlo/$m.py" ] || echo "$m"; done)
[ -z "$missing" ] || echo "WARNING: imported but not in the release (add to DLO in sync.sh): $missing" >&2
echo "synced $(echo $DLO | wc -w) dlo modules and $(echo $SCRIPTS | wc -w) scripts from $SRC"
