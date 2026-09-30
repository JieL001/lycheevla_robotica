#!/bin/bash
# selection track: render evaluation scenes and training sets (5 workers, gated on free commit)
source "$(dirname "$0")/common.sh"
SEL=$DATA/select
rd() {  # rd <dir under $DATA> <render_select.py args...>
  local out=$1; shift
  [ -f $DATA/$out/info.json ] && return 0
  acquire 7
  echo "== render $out $(date)"
  $PY -u scripts/render_select.py "$@" --workers 3 --label_workers 8 --out $DATA/$out 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
}
rd select_eval/iid_600   --split iid --start 0 --n 600 --mode pair
rd select_eval/sal_600   --split saliency_rev --start 0 --n 600 --mode pair
rd select/paired         --split train --start 0 --n 4000 --mode pair
rd select/rho00          --split train --rho 0 --start 0 --n 8000 --mode single
rd select/rho90          --split train --rho 0.9 --start 0 --n 8000 --mode single
rd select/indep          --split train --start 0 --n 4000 --mode indep
rd select/rho50          --split train --rho 0.5 --start 0 --n 8000 --mode single
rd select_eval/occ_300   --split occ_ood --start 0 --n 300 --mode pair
rd select_eval/dens_300  --split density_ood --start 0 --n 300 --mode pair
rd select_eval/lang_300  --split lang_ood --start 0 --n 300 --mode pair
rd select_eval/attr_300  --split attr_ood --start 0 --n 300 --mode pair
echo "SELECT_DATA_DONE $(date)"
