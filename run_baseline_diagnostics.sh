#!/usr/bin/env bash
# Targeted baseline diagnostics; delegates measurements to run_large.sh.
set -euo pipefail
export NPBENCH_RESOURCE_POLICY="${NPBENCH_RESOURCE_POLICY:-fixed}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: ./run_baseline_diagnostics.sh [--dry-run] [GROUP] [RESULT_DIRECTORY]
Groups: all (default), correctness, contracts, restore, performance, p2,
        large_spots (opt-in; never included in all).
Activate the prepared NPBench environment first. No dependencies are installed.
Default all: 51 selected S cells; no CoVA. Original work and tolerances preserved.
correctness: 6 repeats + 1 fresh process; restore: 1 repeat + 1 fresh process;
performance: 2 repeats + 2 fresh processes; contracts/p2: 1 repeat, no fresh process.
Default timeout: 240 seconds per worker (NPBENCH_TIMEOUT overrides).
Missing S goldens may be created once; large_spots requires existing L goldens.
Large spots use 6 repeats/1 fresh process for cholesky and syrk, and
1 repeat/1 fresh process for the other three selected implementations.
Direct saved-artifact checks follow correctness; metadata-only inspection
follows restore. These prohibit compilation and are not performance samples.
Use the same output directory to resume finished steps, including failures.
--dry-run only prints the exact plan; does not run benchmarks or touch results.
Failures are expected observations. The script continues and exits 1 if any
step failed/was partial. See baseline-diagnostics.md for interpretation.
HELP
    exit 0
fi
dry=false
if [[ "${1:-}" == --dry-run ]]; then dry=true; shift; fi
[[ $# -le 2 ]] || { echo 'Too many arguments' >&2; exit 2; }
group="${1:-all}"
case "$group" in all) groups=(correctness contracts restore performance p2);;
    correctness|contracts|restore|performance|p2|large_spots) groups=("$group");;
    *) echo 'Unknown group; see --help' >&2; exit 2;; esac
python="${NPBENCH_PYTHON:-python}"
if "$dry"; then
    for selected in "${groups[@]}"; do
        echo "Group: $selected"
        "$python" "$repo/scripts/baseline_diagnostic_plan.py" "$selected"
    done
    exit 0
fi
out="$(realpath -m -- "${2:-$repo/.cache/baseline-diagnostics/$(date -u +%Y%m%dT%H%M%SZ)-$group}")"
mkdir -p "$repo/.cache" "$out"
exec 8>"$repo/.cache/baseline-diagnostics.lock"
flock -n 8 || { echo 'Another diagnostic collection is active' >&2; exit 1; }
export NPBENCH_THREADS="${NPBENCH_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPBENCH_RESOURCE_PROBE=1
# shellcheck source=scripts/native-env.sh
source "$repo/scripts/native-env.sh"
export OMP_PROC_BIND=true OMP_PLACES=cores
export NPBENCH_TIMEOUT="${NPBENCH_TIMEOUT:-240}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
unset NPBENCH_RETRY_FROM NPBENCH_BENCHMARKS NPBENCH_FRAMEWORKS
failed=0
for selected in "${groups[@]}"; do
    folder="$out/$selected"
    mkdir -p "$folder"
    plan="$("$python" "$repo/scripts/baseline_diagnostic_plan.py" "$selected")"
    if [[ -f "$folder/cases.tsv" ]]; then
        [[ "$(cat "$folder/cases.tsv")" == "$plan" ]] || { echo 'Plan changed; use a new result directory' >&2; exit 1; }
    else
        printf '%s\n' "$plan" > "$folder/cases.tsv"
    fi
    export NPBENCH_CASES_FILE="$folder/cases.tsv" NPBENCH_PRESET=S
    case "$selected" in
        correctness) export NPBENCH_REPEATS=6 NPBENCH_FRESH_PROCESSES=1;;
        restore) export NPBENCH_REPEATS=1 NPBENCH_FRESH_PROCESSES=1;;
        performance) export NPBENCH_REPEATS=2 NPBENCH_FRESH_PROCESSES=2;;
        contracts|p2) export NPBENCH_REPEATS=1 NPBENCH_FRESH_PROCESSES=0;;
        large_spots) export NPBENCH_PRESET=L NPBENCH_REPEATS=1 NPBENCH_FRESH_PROCESSES=1;;
    esac
    # Large correctness cases receive enough repeats to revisit the historical
    # call-3/call-5 failures; each subset has its own frozen collection contract.
    if [[ "$selected" == large_spots ]]; then
        head -n 2 "$folder/cases.tsv" > "$folder/correctness.tsv"
        tail -n 3 "$folder/cases.tsv" > "$folder/performance.tsv"
        for subset in correctness performance; do
            export NPBENCH_CASES_FILE="$folder/$subset.tsv"
            if [[ "$subset" == correctness ]]; then export NPBENCH_REPEATS=6; else export NPBENCH_REPEATS=1; fi
            if ! NPBENCH_REQUIRE_GOLDEN=1 bash "$repo/run_large.sh" baselines "$folder/$subset"; then failed=1; fi
        done
        continue
    fi
    if ! bash "$repo/run_large.sh" baselines "$folder/collection"; then failed=1; fi
    if [[ "$selected" == correctness || "$selected" == restore ]]; then
        if [[ "$selected" == correctness ]]; then benches=(cholesky syrk); extra=(); else benches=(channel_flow softmax); extra=(--inspect-only); fi
        for benchmark in "${benches[@]}"; do
            probe="$folder/direct-$benchmark"
            mkdir -p "$probe"
            if [[ -f "$probe/exit-code.txt" ]]; then
                [[ "$(cat "$probe/exit-code.txt")" == 0 ]] || failed=1
                continue
            fi
            # Take the same checkout-wide lock as the measurement runner.
            if (
                flock -n 9 || exit 1
                taskset --cpu-list "$("$python" -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["affinity"])))' "$folder/collection/collection.json")" \
                    timeout --kill-after=10s "${NPBENCH_TIMEOUT}s" env CUPY_CACHE_DIR="$probe/cupy-cache" "$python" "$repo/scripts/probe_dace_artifact.py" \
                    --collection "$folder/collection" --benchmark "$benchmark" --output "$probe" "${extra[@]}"
            ) 9>"$repo/.cache/large-run.lock" > "$probe/stdout.log" 2> "$probe/stderr.log"; then rc=0; else rc=$?; failed=1; fi
            printf '%s\n' "$rc" > "$probe/exit-code.txt"
        done
    fi
done
echo "Diagnostics finished: $out (failure/partial flag: $failed)"
exit "$failed"
