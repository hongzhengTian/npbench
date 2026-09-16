#!/usr/bin/env bash
# Retry only original failures affected by the protocol-3 infrastructure fixes.
set -euo pipefail
export NPBENCH_RESOURCE_POLICY="${NPBENCH_RESOURCE_POLICY:-fixed}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: ./retry_affected.sh [--dry-run] [OLD_COLLECTION] [NEW_RESULT_DIRECTORY]
Default source: .cache/large-runs/large-20260912; default preset/resources/repeats:
L, 2 CPU threads, 3 fresh processes, 5 subsequent calls per process.
Activate the prepared environment first. --dry-run reads receipts only.
Selects only failed DaCe cache-location/scalar-ABI and Numba cache-capability
cells, never CoVA, complete successes, or unrelated upstream failures.
Uses run_large.sh for collection, locking, provenance and resume. Goldens are
required, never regenerated. Old results remain untouched. Use the same NEW
result directory to resume; finished retry cells (including failures) are skipped.
A selected failed cell needs a new initialization to create isolated artifacts;
its previous successful calls are retained in the OLD collection.
NPBench/benchmark failures can remain, and memory-only Numba cannot gain durable
reuse. Inspect cell receipts/manifests; the aggregate script exits nonzero for
partial/failed cells. run_large.sh environment overrides remain available.
HELP
    exit 0
fi
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
[[ $# -le 2 ]] || { echo 'Too many arguments; see --help' >&2; exit 2; }
export NPBENCH_RETRY_FROM
NPBENCH_RETRY_FROM="$(realpath -e -- "${1:-$repo/.cache/large-runs/large-20260912}")"
if "$dry_run"; then
    exec "${NPBENCH_PYTHON:-python}" "$repo/scripts/retry_affected.py" "$NPBENCH_RETRY_FROM"
fi
export NPBENCH_THREADS="${NPBENCH_THREADS:-2}" NPBENCH_REPEATS="${NPBENCH_REPEATS:-5}" NPBENCH_FRESH_PROCESSES="${NPBENCH_FRESH_PROCESSES:-3}"
export NPBENCH_REQUIRE_GOLDEN=1
out="${2:-$repo/.cache/large-runs/$(date -u +%Y%m%dT%H%M%SZ)-affected-retry}"
exec bash "$repo/run_large.sh" baselines "$out"
