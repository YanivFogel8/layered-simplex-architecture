#!/usr/bin/env bash
# Reproduce the paper's results, cheapest first.
#
#   bash reproduce.sh --smoke      # every result at reduced scale (~20 min)
#   bash reproduce.sh              # the full published runs (many hours)
#   bash reproduce.sh --dry-run    # print the plan only
#   bash reproduce.sh fig2         # one result id (see results_manifest.json)
#
# Each command is timed and logged separately; a failure is recorded and
# stepped over so one broken run cannot abort the sweep.  Summary at the
# end.  JOBS sets the worker count (default: all cores).

set -u -o pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
JOBS="${JOBS:-$($PY -c 'import os; print(os.cpu_count() or 1)')}"
LOGDIR="${LOGDIR:-output/reproduce_logs}"
MODE="full"
WHICH="all"
for a in "$@"; do
  case "$a" in
    --smoke) MODE="smoke" ;;
    --dry-run) MODE="dry" ;;
    validate|fig1|corpus|table7|table8|fig2|fig5|benchmark|scaling|all) WHICH="$a" ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
[ -f "$SUMMARY" ] || printf 'name\tstatus\tseconds\tlog\n' > "$SUMMARY"

run () {                      # run <name> <command...>
  local name="$1"; shift
  local log="$LOGDIR/$name.log"
  if [ "$MODE" = "dry" ]; then printf '  %-22s %s\n' "$name" "$*"; return; fi
  printf '[%s] %-22s ' "$(date +%H:%M:%S)" "$name"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    printf 'ok    %6ss\n' "$((SECONDS - t0))"
    printf '%s\tok\t%s\t%s\n' "$name" "$((SECONDS - t0))" "$log" >> "$SUMMARY"
  else
    printf 'FAIL  %6ss  -> %s\n' "$((SECONDS - t0))" "$log"
    printf '%s\tFAIL\t%s\t%s\n' "$name" "$((SECONDS - t0))" "$log" >> "$SUMMARY"
    tail -3 "$log" | sed 's/^/      /'
  fi
}

want () { [ "$WHICH" = "all" ] || [ "$WHICH" = "$1" ]; }

echo "mode: $MODE   jobs: $JOBS   logs: $LOGDIR"

# --- validation gate (Appendix C): always first -------------------------
if want validate; then
  if [ "$MODE" = "smoke" ]; then
    run validate "$PY" scripts/validate_appendix_c.py --quick
  else
    run validate "$PY" scripts/validate_appendix_c.py
  fi
fi

# --- cheap items --------------------------------------------------------
if want fig1; then
  run fig1 "$PY" scripts/fig1_prior_draws.py --d 24 --l 4 --out output/fig1
fi
if want corpus || want table7 || want table8; then
  run corpus "$PY" scripts/get_kjv.py
fi

# --- the Bible (Table 7 / Figure 7) ------------------------------------
if want table7; then
  run bible_baselines "$PY" scripts/bible_baselines_experiment.py \
    --corpus data/kjv.txt --d 100000 \
    --checkpoints 10000,30000,100000,300000,all --out output/bible_baselines
  if [ "$MODE" = "smoke" ]; then
    run bible_lsa "$PY" scripts/unigram_experiment.py --corpus data/kjv.txt \
      --d 100000 --checkpoints 10000 --jobs "$JOBS" --out output/table7_bible
  else
    run bible_lsa "$PY" scripts/unigram_experiment.py --corpus data/kjv.txt \
      --d 100000 --checkpoints 10000,30000,100000,300000,all \
      --jobs "$JOBS" --out output/table7_bible
  fi
  run bible_report "$PY" scripts/bible_report.py \
    --unigram output/table7_bible/results.json \
    --baselines output/bible_baselines/results.json --out output/bible_report
fi

# --- Table 8 (order-one Bible) ------------------------------------------
if want table8; then
  if [ "$MODE" = "smoke" ]; then
    run table8_smoke "$PY" scripts/state_family_experiment.py \
      --corpus data/kjv.txt --n 20000 --d 100000 --m-grid 0,4,16 \
      --l-max 20 --jobs "$JOBS" --out output/table8_smoke
  else
    run table8_m256 "$PY" scripts/state_family_experiment.py \
      --corpus data/kjv.txt --d 100000 --m-grid 0,64,128,256 \
      --l-max 40 --jobs "$JOBS" --out output/table8_m256
    run table8_m512 "$PY" scripts/state_family_experiment.py \
      --corpus data/kjv.txt --d 100000 --m-grid 0,64,128,256,512 \
      --l-max 60 --jobs "$JOBS" --out output/table8_m512
  fi
fi

# --- Figure 2 / Table 1 -------------------------------------------------
if want fig2; then
  if [ "$MODE" = "smoke" ]; then
    run fig2_smoke "$PY" scripts/depth_tilt_experiment.py --d 1000 --n 316 \
      --l-max 69 --alphas 0,1.5,3.0 --profiles 8 --jobs "$JOBS" \
      --out output/fig2_d1e3
    run fig2_report "$PY" scripts/depth_tilt_report.py \
      --results output/fig2_d1e3/results.json --out output/fig2_report
  else
    run fig2_d1e3 "$PY" scripts/depth_tilt_experiment.py --d 1000 --n 316 \
      --l-max 69 --jobs "$JOBS" --out output/fig2_d1e3
    run fig2_d1e4 "$PY" scripts/depth_tilt_experiment.py --d 10000 --n 1000 \
      --l-max 92 --jobs "$JOBS" --out output/fig2_d1e4
    run fig2_d1e6 "$PY" scripts/depth_tilt_experiment.py --d 1000000 --n 1000 \
      --l-max 138 --jobs "$JOBS" --out output/fig2_d1e6
    run fig2_report "$PY" scripts/depth_tilt_report.py \
      --results output/fig2_d1e3/results.json output/fig2_d1e4/results.json \
                output/fig2_d1e6/results.json --out output/fig2_report
  fi
fi

# --- Figure 5 / Table 4 -------------------------------------------------
if want fig5; then
  if [ "$MODE" = "smoke" ]; then
    run fig5 "$PY" scripts/depth_scaling_experiment.py --d 2000 --n 200 \
      --alphas 2,3 --c-values 0.5,1,1.5,2,2.37,3,4 --profiles 6 \
      --description-draws 100 --jobs "$JOBS" --out output/fig5
  else
    run fig5 "$PY" scripts/depth_scaling_experiment.py --d 100000 --n 1000 \
      --alphas 2,3,4 --jobs "$JOBS" --out output/fig5
  fi
fi

# --- Figure 6 / Tables 5-6 ----------------------------------------------
if want benchmark; then
  if [ "$MODE" = "smoke" ]; then
    run benchmark "$PY" scripts/benchmark_experiment.py \
      --out output/benchmark --d 300 --n-values 100,300,1000 --trials 3 \
      --l-max 40 --targets uniform,zipf_1.5,zipf_5,dirichlet_half \
      --jobs "$JOBS"
  else
    run benchmark "$PY" scripts/benchmark_experiment.py \
      --out output/benchmark --jobs "$JOBS"
  fi
  run benchmark_report "$PY" scripts/benchmark_report.py \
    --results output/benchmark/results.json --out output/benchmark
fi

# --- Figures 3-4 / Tables 2-3 (the heavy factorial grid, last) ----------
if want scaling; then
  if [ "$MODE" = "smoke" ]; then
    run factorial "$PY" scripts/factorial_scaling_experiment.py \
      --d-values 1000,3162 --N-values 100,316,1000 --alphas 2,3 \
      --profile-samples 60 --workers "$JOBS" \
      --out-csv output/factorial/factorial_results.csv
    run scaling_report "$PY" scripts/scaling_report.py \
      --results output/factorial/factorial_results.csv \
      --out output/scaling --discovery-draws 300 --description-draws 200
  else
    run factorial "$PY" scripts/factorial_scaling_experiment.py \
      --d-values 1000,3162,10000,31623,100000 \
      --N-values 100,316,1000,3162,10000 --alphas 1.5,2,3,4 \
      --c-value cstar --profile-samples 5000 --workers "$JOBS" \
      --out-csv output/factorial/factorial_results.csv
    run scaling_report "$PY" scripts/scaling_report.py \
      --results output/factorial/factorial_results.csv --out output/scaling
  fi
fi

if [ "$MODE" = "dry" ]; then exit 0; fi
echo
echo "=== summary ==="
column -t -s "$(printf '\t')" "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
nfail=$(awk -F'\t' '$2=="FAIL"' "$SUMMARY" | wc -l | tr -d ' ')
echo "$nfail failed; logs in $LOGDIR"
