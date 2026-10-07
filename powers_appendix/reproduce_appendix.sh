#!/usr/bin/env bash
# Current main-paper protocol. Historical results in the supplied ZIP are not reused.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
PROCS=${PROCS:-6}
OUT=${OUT:-results_main}
mkdir -p "$OUT/bench" "$OUT/bible"
"$PY" table1_trials.py --out "$OUT/bench/trials.pkl"
for prior in delta exp1 unif:2 unif:1; do
  "$PY" run_bench.py "$prior" --out "$OUT/bench" --Lmax 80 --procs "$PROCS"
done
"$PY" run_power.py --K 80 --trials "$OUT/bench/trials.pkl" --out "$OUT/bench" --procs "$PROCS"
"$PY" table1_build.py --out "$OUT/bench"
for prior in delta exp1 unif:2 unif:1; do
  "$PY" run_bible.py "$prior" --out "$OUT/bible" --Lmax 54 --du 0.0125 --procs "$PROCS"
done
"$PY" run_bible_power.py --out "$OUT/bible" --procs "$PROCS" --K 27
"$PY" table2_build.py --out "$OUT/bible"
"$PY" tables_tex.py --out "$OUT"
