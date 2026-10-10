#!/usr/bin/env python3
"""CPU test of the prefill3 routed-MoE overlay (overlay/glm_pf3_moe.py + glm_pf3_jit.py) as merged into final3.
No GPU and no vLLM: the vLLM hook targets are stand-in modules.

  A. JIT sources (the glm-prefill3-20260928 test_pf3_cpu.py checks, run against the OVERLAY's glm_pf3_jit): the act
     patch is reversible to the vendored vLLM Marlin template, launch tables match the instance plan, and the gate_up
     permutation is a 64-column-block permutation.
  B. The served kernel pair (TILE, act and plain) is in the instance plan the JIT builds.
  C. Hook plumbing: switches off -> register() is inert; switches on -> the four hooks install on already-imported
     modules and are idempotent; every wrapper passes through to stock when its switch is off (per call, glm_ab), when
     the call is ineligible, or when the extension failed to build; _apply_quant_method sets / restores the runner
     context; _unpack passes through when nothing is pending and fails stop on a broken pointer handshake.
  D. sitecustomize registers the pf3 block after the trunc block and before the gather-route block.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OVL = os.path.join(REPO, "overlay")
sys.path.insert(0, OVL)
# default_instances() (the microbench plan of part A) needs the Marlin smem model in tests/pf3_model/; the served
# path never calls it (ext() builds exactly TILE, act and plain), so the model is not part of the overlay.
sys.path.insert(0, os.environ.get("GLM_PF3_VENDOR", os.path.join(HERE, "pf3_model")))
FAIL = []


def check(cond, what):
    print(("[PASS] " if cond else "[FAIL] ") + what)
    if not cond:
        FAIL.append(what)


# ---------------------------------------------------------------------------------------------------------------- A
import glm_pf3_jit as PJ  # noqa: E402

inst = PJ.default_instances()
plain = [i for i in inst if not i["act"]]
act = [i for i in inst if i["act"]]
check(any(i == dict(tmb=4, tk=64, tn=256, th=256, stages=4, act=False) for i in inst), "stock tile in the plan")
texts = PJ.render_texts(inst)
tpl = texts["marlin_moe_template_pf3.h"]
check(tpl.count("#if GLM_PF3_ACT") == 3 and "glm_pf3_act_int4(sh_red[c_sh_rd]" in tpl, "act epilogue patch present")
orig = PJ.read_vendored()[PJ.MOE_TEMPLATE]
back = tpl[len(PJ.REMAP_DEFS):].replace(PJ.ACT_PATCH, PJ.ACT_ANCHOR)
for old, new, cnt in PJ.REMAP_EDITS:
    back = back.replace(new, old)
check(back == orig, "patch is reversible to the vendored vLLM template")
check(tpl.count("GLM_PF3_BCOL(slice_col)") == 2 and tpl.count("GLM_PF3_SCOL(slice_col)") == 2, "column remap edits")
check(texts["launch_plain.cu"].count("&glm_pf3_moe::") == len(plain), "launch_plain rows == plain instances")
check(texts["launch_act.cu"].count("&glm_pf3_moe_act::") == len(act), "launch_act rows == act instances")
ok = True
for n, t in texts.items():
    if n.startswith("moe_act_"):
        ok &= "#define GLM_PF3_ACT 1" in t
    if n.startswith("moe_plain_"):
        ok &= "#define GLM_PF3_ACT 0" in t
check(ok, "act / plain translation units define GLM_PF3_ACT accordingly")
ok = True
for tn in (128, 256):
    p = PJ.gate_up_perm(1024, tn)
    ok &= sorted(p) == list(range(1024))
    h = tn // 2
    for j in range(1024 // tn):
        ok &= p[j * tn:j * tn + h] == list(range(j * h, (j + 1) * h))
        ok &= p[j * tn + h:(j + 1) * tn] == list(range(512 + j * h, 512 + (j + 1) * h))
    blocks = p[::64]
    ok &= all(p[b * 64:(b + 1) * 64] == list(range(blocks[b], blocks[b] + 64)) for b in range(len(blocks)))
check(ok, "gate_up permutation: 64-column blocks, gate/up halves per tile")
vc = open(os.path.join(OVL, "glm_pf3_csrc", "VLLM_COMMIT")).read().strip()
check(bool(vc), f"vendored headers pinned: {vc[:60]}")

# ---------------------------------------------------------------------------------------------------------------- C
for k in ("GLM_PF3_SUMADD", "GLM_PF3_DOWN", "GLM_PF3_ACT"):
    os.environ.pop(k, None)
ab = types.ModuleType("glm_ab")
ab.ACTIVE, ab.values = False, {}
ab.env = lambda name, default=None: ab.values.get(name, default)
sys.modules["glm_ab"] = ab
import glm_pf3_moe as M  # noqa: E402

# B. the served kernel pair is built by ext() from TILE (+act): both must be launchable instances of the JIT
want = [dict(M.TILE, act=False), dict(M.TILE, act=True)]
check(all(w in inst for w in want), f"served tile {M.TILE} (act and plain) is in the JIT instance plan")
tw = PJ.render_texts(want)
check(tw["launch_plain.cu"].count("&glm_pf3_moe::") == 1 and tw["launch_act.cu"].count("&glm_pf3_moe_act::") == 1,
      "the served pair renders one plain and one act launcher row")

n_meta = len(sys.meta_path)
M.register()
check(len(sys.meta_path) == n_meta and not M.installed(), "switches off: register() is inert")

calls = []
marlin = types.ModuleType(M.MARLIN_MOD)


def _fused_marlin_moe(hidden_states, w1=None, w2=None, block_size_m=64, quant_type=None, output=None, **kw):
    calls.append(("fused", hidden_states))
    return "stock-fused"


class MarlinExperts:
    def moe_sum(self, input, output, topk_ids, expert_map):
        calls.append(("sum", input))
        return "stock-sum"


marlin._fused_marlin_moe, marlin.MarlinExperts = _fused_marlin_moe, MarlinExperts
runner_mod = types.ModuleType(M.RUNNER_MOD)
runner_mod._unpack = lambda result: result


class MoERunner:
    def _apply_quant_method(self, *a, **kw):
        return getattr(M._TLS, "runner", None)


runner_mod.MoERunner = MoERunner
sys.modules[M.MARLIN_MOD], sys.modules[M.RUNNER_MOD] = marlin, runner_mod
os.environ.update(GLM_PF3_SUMADD="1", GLM_PF3_DOWN="1", GLM_PF3_ACT="1")
M.register()
f1, s1, u1, q1 = marlin._fused_marlin_moe, MarlinExperts.moe_sum, runner_mod._unpack, MoERunner._apply_quant_method
check(all(getattr(f, "_glm_pf3", False) for f in (f1, s1, u1, q1)), "switches on: four hooks installed")
M.register()
check((marlin._fused_marlin_moe, MarlinExperts.moe_sum, runner_mod._unpack, MoERunner._apply_quant_method)
      == (f1, s1, u1, q1), "re-register is idempotent (no double wrap)")

import torch  # noqa: E402

# per-call switch off through glm_ab
ab.ACTIVE, ab.values = True, {"GLM_PF3_SUMADD": "0", "GLM_PF3_DOWN": "0", "GLM_PF3_ACT": "0"}
hs = torch.zeros((4096, 64), dtype=torch.bfloat16)
check(marlin._fused_marlin_moe(hs) == "stock-fused", "glm_ab variant off: fused_marlin_moe -> stock")
check(MarlinExperts().moe_sum(torch.zeros((2048, 8, 64), dtype=torch.bfloat16), None, None, None) == "stock-sum",
      "glm_ab variant off: moe_sum -> stock")
ab.values = {"GLM_PF3_SUMADD": "1", "GLM_PF3_DOWN": "1", "GLM_PF3_ACT": "1"}
# ineligible calls (small M; decode sizes) and no runner context stay stock
check(marlin._fused_marlin_moe(torch.zeros((16, 64), dtype=torch.bfloat16)) == "stock-fused",
      "variant on, M = 16: fused_marlin_moe -> stock")
check(MarlinExperts().moe_sum(torch.zeros((2048, 8, 64), dtype=torch.bfloat16), torch.zeros((2048, 64)), None, None)
      == "stock-sum", "variant on, no runner context: moe_sum -> stock")
# extension failure: every path stock (CPU torch has no CUDA stream to ask: not capturing)
M._capturing = lambda: False
M.S["failed"] = "test: no nvcc"
check(marlin._fused_marlin_moe(hs) == "stock-fused", "extension failed: fused_marlin_moe -> stock")
r = MoERunner()
M._TLS.runner = r
check(MarlinExperts().moe_sum(torch.zeros((2048, 8, 64), dtype=torch.bfloat16), torch.zeros((2048, 64)), None, None)
      == "stock-sum", "extension failed: moe_sum -> stock")
M._TLS.runner = None
# runner context
check(MoERunner()._apply_quant_method() is not None and getattr(M._TLS, "runner", None) is None,
      "_apply_quant_method sets the runner context for the call and restores it")
# _unpack
sh, fu = torch.zeros(4), torch.ones(4)
check(runner_mod._unpack((sh, fu)) == (sh, fu), "_unpack passes through when nothing is pending")
M._TLS.pending = fu.data_ptr() + 1
try:
    runner_mod._unpack((sh, fu))
    check(False, "_unpack fails stop on a broken pointer handshake")
except RuntimeError:
    check(True, "_unpack fails stop on a broken pointer handshake")
M._TLS.pending = fu.data_ptr()
check(runner_mod._unpack((sh, fu)) == (None, fu), "_unpack after a fused sum-add hands back (None, fused)")
ab.ACTIVE = False

# ---------------------------------------------------------------------------------------------------------------- D
sc = open(os.path.join(OVL, "sitecustomize.py")).read()
pos = [sc.find(k) for k in ("glm_draft_trunc.register()", "glm_pf3_moe.register()", "glm_roce_gather_route.register()")]
check(all(p > 0 for p in pos) and pos == sorted(pos), f"sitecustomize order trunc < pf3 < gather-route {pos}")
for k in ("GLM_PF3_SUMADD", "GLM_PF3_DOWN", "GLM_PF3_ACT"):
    check(f'"{k}"' in sc, f"sitecustomize knows {k} (install gate + glm_ab key)")

print(f"ALL {'PASS' if not FAIL else 'FAIL'}: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
