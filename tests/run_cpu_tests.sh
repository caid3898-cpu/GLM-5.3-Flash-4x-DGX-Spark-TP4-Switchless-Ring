#!/usr/bin/env bash
# Every CPU test of the stack, no GPU. Run from anywhere:
#   tests/run_cpu_tests.sh [LOGDIR]          -> LOGDIR/cpu_tests.txt (one PASS / FAIL / SKIP line per suite) + logs
# Python: uv (torch, numpy, safetensors, pytest, tqdm pulled on the fly) when it is installed, else python3 with those
# packages. Suites that compare against the image's vLLM sources use GLM_IMAGE_SRC (an extracted source tree whose
# vllm/ is the image's) or the vLLM installed next to the interpreter; without either they print a skip.
#   GLM_VLLM_SRC   vLLM 487ecf187 tree that imports on CPU: also runs tests/test_glm_prefill_sched.py (else SKIP)
#   GLM_SWEEP_DIR  the dense-fast sweep measurements: also regenerates overlay/glm_dense_fast_table.json
# The *_gpu.py / gpu_test_*.py files need a free GPU (model stopped) and are not run here; test_glm_ab_image.py runs
# inside the serving image (see its header).
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
LOG=${1:-$R/tests/.cpu_logs}; mkdir -p "$LOG"; : > "$LOG/cpu_tests.txt"
if command -v uv >/dev/null 2>&1; then
  PY=(uv run --quiet --no-project --with torch --with numpy --with safetensors --with pytest --with tqdm python)
else
  PY=(python3)
fi
run(){ local name=$1; shift; local t0=$SECONDS
  ( cd "$R" && "$@" ) > "$LOG/$name.log" 2>&1; local rc=$?
  printf '%-34s %s  (%3d s)  %s\n' "$name" "$([[ $rc == 0 ]] && echo PASS || echo "FAIL rc=$rc")" $((SECONDS - t0)) \
    "$(grep -aE 'ALL|passed|failed|PASS|FAIL|OK|skipped' "$LOG/$name.log" | tail -1 | cut -c1-90)" | tee -a "$LOG/cpu_tests.txt"; }

for t in test_glm_ab test_glm_fast_load test_glm_prefill_hooks test_glm_prefill_shard test_glm_dense_fast \
         test_glm_draft_conv_fused test_glm_l2_prefetch_mla test_glm_marlin_tune_overlay test_glm_mhc_bf16w \
         test_proxy_pin test_speedscreen_ab_gates test_glm_cert_math test_glm_cert_head_mintok \
         test_glm_draft_trunc test_glm_gumbel_coupled test_trunc_gumbel_compose test_trunc_cost_c4fit \
         test_glm_draft_trunc_v3 test_glm_devselect \
         test_glm_flashkda test_glm_kda_conv_split test_glm_sparse_mla_prefill test_glm_pf3_cpu \
         test_glm_roce_gather_route test_final_bench_cpu; do
  run "$t" "${PY[@]}" tests/$t.py
done
run test_glm_roce_cpu "${PY[@]}" roce/tests/test_glm_roce_cpu.py
run test_glm_mamba_align_fix "${PY[@]}" -B tests/test_glm_mamba_align_fix.py
run test_prefix_scan_cpu bash -c "cd bench && python3 -B -S -m unittest test_prefix_scan_cpu"
run pytest_kda_stash_mhc_runtime "${PY[@]}" -m pytest -q -p no:cacheprovider tests/test_kda_stash_boundary.py \
  tests/test_mhc_fused.py tests/test_runtime_package.py tests/test_switchless.py
if [[ -n ${GLM_VLLM_SRC:-} ]]; then
  run test_glm_prefill_sched "${PY[@]}" tests/test_glm_prefill_sched.py
else
  echo "test_glm_prefill_sched             SKIP  (set GLM_VLLM_SRC to a vLLM 487ecf187 tree that imports on CPU)" | tee -a "$LOG/cpu_tests.txt"
fi

# Every overlay module parses, and sitecustomize registers the hook blocks in the order they were validated in.
run overlay_registration_order "${PY[@]}" - <<'PY'
import ast, glob, os
ovl = "overlay"
for f in glob.glob(f"{ovl}/*.py"):
    ast.parse(open(f).read(), f)
s = open(f"{ovl}/sitecustomize.py").read()
order = ["glm_cert_head.register", "glm_early_plan.register", "glm_marlin_tune.register", "glm_prefill_hooks.register",
         "glm_l2_prefetch_mla.register", "glm_mhc_bf16w.register", "glm_kda_conv_split.register",
         "glm_flashkda.register", "glm_sparse_mla_prefill.register", "glm_gumbel_coupled.register",
         "glm_draft_trunc.register", "glm_pf3_moe.register", "glm_roce_gather_route.register", "glm_l2pf_v2.register"]
pos = [s.find(k) for k in order]
assert all(p > 0 for p in pos) and pos == sorted(pos), dict(zip(order, pos))
for j in ("glm_bav_table_seg.json", "glm_draft_trunc_cost.json", "glm_draft_trunc_cost_c4fit.json", "glm_devselect.py", "glm_l2pf_v2.py",
          "glm_marlin_tune.tuned.json", "glm_dense_fast_table.json", "glm_pf3_csrc/VLLM_COMMIT",
          "glm_flashkda_csrc/flash_kda.cpp", "glm_dense_fast_csrc/glmk_dense.cu"):
    assert os.path.exists(f"{ovl}/{j}"), j
assert os.path.exists("profiles/levers_policy.json")
print("OK overlay parses; registration order", " -> ".join(k.split(".")[0] for k in order))
PY

n=$(grep -c ' PASS ' "$LOG/cpu_tests.txt"); f=$(grep -c ' FAIL' "$LOG/cpu_tests.txt"); s=$(grep -c ' SKIP ' "$LOG/cpu_tests.txt")
echo "CPU TESTS: $n pass, $f fail, $s skip" | tee -a "$LOG/cpu_tests.txt"
[[ $f == 0 ]]
