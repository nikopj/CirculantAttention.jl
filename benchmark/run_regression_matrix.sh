#!/usr/bin/env bash
#
# Narrow down the CUDA.jl reduction-kernel regression across the axes that
# changed between Julia 1.11 (old) and 1.12 (new):
#   - Julia/LLVM   : 1.11/LLVM16  vs  1.12/LLVM18   (bound together)
#   - CUDA.jl      : 5.4          vs  CUDACore 6.2  (bound to the projects)
#   - CUDA runtime : 12.5         vs  12.9          (ptxas — independently pinnable)
#
# Runs entirely from the command line. Pins the CUDA runtime via
# Preferences (CUDA.set_runtime_version!) to isolate the ptxas/runtime axis
# WITHOUT building new environments. Each project's LocalPreferences.toml is
# snapshotted and restored, so nothing is left modified.
#
# Assumes the two project dirs are siblings in the current directory, each
# containing src/toolchain_regression_mwe.jl. Override the paths below if not.
set -uo pipefail

OLD_PROJ="${OLD_PROJ:-RegressionMWE_111}"   # Julia 1.11 / CUDA.jl 5.4
NEW_PROJ="${NEW_PROJ:-RegressionMWE_112}"   # Julia 1.12 / CUDACore 6.2
OLD_JL="${OLD_JL:-1.11}"
NEW_JL="${NEW_JL:-1.12}"
OLD_RT="${OLD_RT:-12.5}"                     # runtime each project ships with
NEW_RT="${NEW_RT:-12.9}"
OUT="${OUT:-mwe_matrix_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT"
echo "results -> $OUT"

# ---------------------------------------------------------------- pin idle GPU
# This node is shared: HBM contention makes single-run dist times unreliable
# (the saxpy control drifts run-to-run). Pin every config to the same idlest
# device so cross-config comparisons are apples-to-apples. Respect a
# pre-set CUDA_VISIBLE_DEVICES if the caller already chose one.
pick_idle_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2,2n -k3,3nr | head -1 | cut -d, -f1 | tr -d ' '
}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(pick_idle_gpu)}"
echo "pinned to GPU $CUDA_VISIBLE_DEVICES (lowest utilization, most free mem)"

# ------------------------------------------------------------------ prefs guard
# snapshot LocalPreferences.toml for both projects; restore on any exit
declare -A PREF_HAD
snapshot_prefs() {
    local p="$1"
    if [ -f "$p/LocalPreferences.toml" ]; then
        cp "$p/LocalPreferences.toml" "$OUT/.$(basename "$p").prefbak"
        PREF_HAD["$p"]=1
    else
        PREF_HAD["$p"]=0
    fi
}
restore_prefs() {
    local p="$1"
    if [ "${PREF_HAD[$p]:-0}" = "1" ]; then
        cp "$OUT/.$(basename "$p").prefbak" "$p/LocalPreferences.toml"
    else
        rm -f "$p/LocalPreferences.toml"     # remove any we created
    fi
}
restore_all() { restore_prefs "$OLD_PROJ"; restore_prefs "$NEW_PROJ"; }
trap restore_all EXIT INT TERM

snapshot_prefs "$OLD_PROJ"
snapshot_prefs "$NEW_PROJ"

# --------------------------------------------------------------------- helpers
set_runtime() {  # proj juliaver version  -> 0 on success
    julia +"$2" --project="$1" -e \
        "using CUDA; CUDA.set_runtime_version!(v\"$3\")" 2>&1 | tail -2
}

run_mwe() {  # juliaver proj outname dump
    local jv="$1" proj="$2" name="$3" dump="$4"
    echo ">>> $name : julia +$jv --project=$proj  (dump=$dump)"
    MWE_OUTFILE="$PWD/$OUT/$name.txt" MWE_DUMP_CODEGEN="$dump" \
        julia +"$jv" --project="$proj" "$proj/src/toolchain_regression_mwe.jl" \
        > "$OUT/$name.console.log" 2>&1
    echo "    exit=$? -> $OUT/$name.txt"
}

collect_dump() {  # proj label
    local proj="$1" label="$2" d
    d=$(ls -dt "$proj"/src/codegen_* 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -d "$d" ]; then
        rm -rf "$OUT/codegen_$label"; mv "$d" "$OUT/codegen_$label"
        echo "    codegen -> $OUT/codegen_$label"
    fi
}

# pick the kernel file (largest of an extension) from a dump dir
biggest() { ls -S "$1"/*."$2" 2>/dev/null | head -1; }

# PTX may be dumped as .ptx or .asm depending on GPUCompiler version
ptx_file() { local d="$1" f; f=$(biggest "$d" ptx); [ -z "$f" ] && f=$(biggest "$d" asm); echo "$f"; }
ptx_fingerprint() {  # dumpdir label
    local f; f=$(ptx_file "$1"); local lab="$2"
    [ -f "$f" ] || { printf "    %-12s (no ptx/asm)\n" "$lab"; return; }
    printf "    %-12s insts=%-5s ld=%-4s st=%-4s fma=%-4s mul=%-4s add=%-4s\n" \
        "$lab" \
        "$(grep -cE '^[[:space:]]+[a-z]' "$f")" \
        "$(grep -c 'ld\.'  "$f")" "$(grep -c 'st\.'  "$f")" \
        "$(grep -c 'fma\.' "$f")" "$(grep -c 'mul\.' "$f")" \
        "$(grep -c 'add\.' "$f")"
}

w45_time()  { awk '$1=="45"{print $3; exit}' "$OUT/$1.txt" 2>/dev/null; }
saxpy_line() { grep 'GB/s' "$OUT/$1.txt" 2>/dev/null | tail -1; }
saxpy_ms()   { grep 'GB/s' "$OUT/$1.txt" 2>/dev/null | tail -1 | awk '{print $1}'; }
ratio()      { awk -v a="$1" -v b="$2" 'BEGIN{ if(b+0>0) printf "%.2f", a/b; else printf "-" }'; }

# ======================================================================== runs
# A: old baseline (1.11) + codegen dump
run_mwe "$OLD_JL" "$OLD_PROJ" "A_old_base"   1; collect_dump "$OLD_PROJ" "old"
# B: new baseline (1.12) + codegen dump
run_mwe "$NEW_JL" "$NEW_PROJ" "B_new_base"   1; collect_dump "$NEW_PROJ" "new"

# C: NEW project, runtime pinned DOWN to old (isolates ptxas/runtime axis)
echo ">>> pinning $NEW_PROJ runtime -> $OLD_RT"
set_runtime "$NEW_PROJ" "$NEW_JL" "$OLD_RT"
run_mwe "$NEW_JL" "$NEW_PROJ" "C_new_rt$OLD_RT" 0
restore_prefs "$NEW_PROJ"

# D: OLD project, runtime pinned UP to new (symmetric; may fail if CUDA.jl 5.4
#    has no artifact for $NEW_RT — failure is itself informative, logged)
echo ">>> pinning $OLD_PROJ runtime -> $NEW_RT"
set_runtime "$OLD_PROJ" "$OLD_JL" "$NEW_RT"
run_mwe "$OLD_JL" "$OLD_PROJ" "D_old_rt$NEW_RT" 0
restore_prefs "$OLD_PROJ"

# ==================================================================== analysis
echo
echo "================ codegen comparison (old 1.11/LLVM16 vs new 1.12/LLVM18) ================"
echo "LLVM IR (.ll) differs across LLVM versions even for identical logic — read the PTX."
for ext in ll ptx asm; do
    f1=$(biggest "$OUT/codegen_old" "$ext"); f2=$(biggest "$OUT/codegen_new" "$ext")
    [ -n "$f1" ] && [ -n "$f2" ] || continue
    if diff -q "$f1" "$f2" >/dev/null 2>&1; then
        echo "  .$ext : IDENTICAL"
    else
        n=$(diff "$f1" "$f2" | grep -cE '^[<>]')
        echo "  .$ext : DIFFERS ($n changed lines)  -> $OUT/diff_$ext.txt"
        diff "$f1" "$f2" > "$OUT/diff_$ext.txt"
    fi
done
echo
echo "PTX fingerprint (instruction mix + register count — localizes the regression to PTX gen):"
ptx_fingerprint "$OUT/codegen_old" "old/LLVM16"
ptx_fingerprint "$OUT/codegen_new" "new/LLVM18"
echo "  (register count is the 'regs' column in the A_old_base.txt / B_new_base.txt tables)"

echo
echo "================= W=45 kernel time, normalized by saxpy (contention-robust) ================="
printf "  %-26s %-10s %-10s %s\n" "config" "W45(ms)" "saxpy(ms)" "W45/saxpy  [the comparable number]"
for cfg in "A_old_base|A old (1.11, rt$OLD_RT)" \
           "B_new_base|B new (1.12, rt$NEW_RT)" \
           "C_new_rt$OLD_RT|C new (1.12, rt$OLD_RT)" \
           "D_old_rt$NEW_RT|D old (1.11, rt$NEW_RT)"; do
    name="${cfg%%|*}"; label="${cfg#*|}"
    w=$(w45_time "$name"); s=$(saxpy_ms "$name")
    printf "  %-26s %-10s %-10s %s\n" "$label" "${w:--}" "${s:--}" "$(ratio "$w" "$s")"
done
echo
echo "interpretation (compare the W45/saxpy column, which factors out HBM contention):"
echo "  C/saxpy ≈ A/saxpy  -> regression is CUDA runtime/ptxas"
echo "  C/saxpy ≈ B/saxpy  -> NOT ptxas; it's Julia/LLVM or CUDACore (see PTX fingerprint)"
echo "  raw saxpy(ms) drifting between configs = the node is contended; trust the ratio, not raw ms"
echo
echo "full result files + dumps under: $OUT/"
