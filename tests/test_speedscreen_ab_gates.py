#!/usr/bin/env python3
"""CPU checks for the two A/B gates added for the 2026-09-28 speed screen (no torch, no GPU):
  * glm_marlin_tune: GLM_MARLIN_TUNE_ON is read per call through glm_ab; the persistent MoE workspace is allocated
    on an eager call of an OFF variant, so an ON variant's capture finds it;
  * glm_roce_proxy_pin: the proxy threads are recorded at start whatever the variant, and glm_ab's switch hook pins
    them for a variant with GLM_ROCE_PROXY_CPUS set and restores the original mask otherwise.
  python3 tests/test_speedscreen_ab_gates.py
"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))


def arm(specs):
    for k in [k for k in os.environ if k.startswith(("GLM_AB", "GLM_MARLIN_TUNE", "GLM_ROCE_PROXY"))]:
        del os.environ[k]
    for m in ("glm_ab", "glm_marlin_tune", "glm_roce_proxy_pin"):
        sys.modules.pop(m, None)
    import glm_ab
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "glm_ab"]
    glm_ab.KNOWN.setdefault("GLM_MARLIN_TUNE_ON", "bool")
    glm_ab.KNOWN.setdefault("GLM_ROCE_PROXY_CPUS", "raw")
    os.environ["GLM_AB_VARIANTS"] = str(len(specs))
    for i, s in enumerate(specs):
        os.environ[f"GLM_AB_V{i}"] = s
    assert glm_ab.configure()
    return glm_ab


def test_marlin_gate():
    ab = arm(["GLM_MARLIN_TUNE_ON=0+GLM_ROCE_PROXY_CPUS=0", "GLM_MARLIN_TUNE_ON=1+GLM_ROCE_PROXY_CPUS=0"])
    assert os.environ["GLM_MARLIN_TUNE_ON"] == "1"          # union: install everywhere, dispatch per call
    import glm_marlin_tune as T
    T.S["table"] = T.Table({"version": 1, "moe": {"persistent_workspace": True, "entries": [
        {"gemm": "gate_up", "m": 4, "thread_k": 128, "thread_n": 128, "blocks_per_sm": 1}]}})
    T.S["moe_ws"].clear()
    capturing = [False]
    T._capturing = lambda: capturing[0]
    T._sms = lambda d: 48
    T._new_workspace = lambda device, n: ("persistent", device, n)
    calls = []

    def orig_gemm(*args, thread_k=-1, thread_n=-1, blocks_per_sm=-1):
        calls.append((thread_k, thread_n, blocks_per_sm))
        return "out"

    gemm = T.make_moe_gemm_wrapper(orig_gemm)
    topk = types.SimpleNamespace(shape=(4, 8))
    scales = types.SimpleNamespace(dtype="torch.float8_e4m3fn")
    args = [None] * 26
    args[4], args[14], args[20], args[21] = scales, topk, 1024, 4096   # b_scales, topk_weights, size_n, size_k

    def call():
        gemm(*args)
        return calls[-1]

    ab.set_runtime(0)
    assert call() == (-1, -1, -1), "variant 0 (ON=0) must run stock"
    ab.set_runtime(1)
    assert call() == (128, 128, 1), "variant 1 (ON=1) must pass the tuned overrides"
    with ab.capturing(0):
        assert call() == (-1, -1, -1), "capture of variant 0 must bake stock"

    mod = types.SimpleNamespace(marlin_make_workspace_new=lambda device, m=1, existing=None: ("stock", device, m))
    T.install_moe(mod)
    ws = mod.marlin_make_workspace_new
    ab.set_runtime(0)
    assert ws("cuda:0", 4)[0] == "stock"
    assert ("cuda:0", False) in {(k[0], k[1]) for k in T.S["moe_ws"]}, "OFF eager call must pre-allocate"
    capturing[0] = True
    with ab.capturing(1):
        assert ws("cuda:0", 4)[0] == "persistent", "ON variant's capture must find the persistent workspace"
    with ab.capturing(0):
        assert ws("cuda:0", 4)[0] == "stock"
    capturing[0] = False
    print("PASS marlin_gate")


def test_proxy_pin_switch():
    ab = arm(["GLM_ROCE_PROXY_CPUS=0+GLM_MARLIN_TUNE_ON=0", "GLM_ROCE_PROXY_CPUS=auto+GLM_MARLIN_TUNE_ON=0"])
    assert os.environ["GLM_ROCE_PROXY_CPUS"] == "1"          # union value must never be read as a CPU list
    import glm_roce_proxy_pin as pp
    tasks = {100}
    masks = {100: set(range(20))}

    fake_os = types.SimpleNamespace(
        environ=os.environ, getpid=lambda: 100,
        sched_getaffinity=lambda tid: set(masks.get(tid or 100, set(range(20)))),
        sched_setaffinity=lambda tid, cpus: masks.__setitem__(tid, set(cpus)),
        listdir=lambda p: [str(t) for t in tasks] if p == "/proc/self/task" else os.listdir(p),
        path=types.SimpleNamespace(exists=lambda p: p.startswith("/proc/self/task/") and int(p.rsplit("/", 1)[1]) in tasks,
                                   join=os.path.join))
    pp.os = fake_os
    pp.big_cores = lambda *a, **k: set(range(5, 10)) | set(range(15, 20))

    class Proxy:
        def start(self):
            tasks.add(200)
            masks[200] = set(range(20))

    mod = types.SimpleNamespace(Proxy=Proxy)
    ab.set_runtime(0)
    pp._install(mod)
    Proxy().start()
    assert masks[200] == set(range(20)), "variant 0 at start: not pinned"
    assert pp.apply_current in ab.SWITCH_HOOKS
    ab.worker_switch(1)
    assert masks[200] == {19}, f"variant 1: pinned to the highest big core, got {masks[200]}"
    assert masks[100] == set(range(20)), "ISOLATE off: other threads untouched"
    ab.worker_switch(0)
    assert masks[200] == set(range(20)), "back to variant 0: original mask restored"
    ab.worker_switch(1)
    assert masks[200] == {19}
    ab.SWITCH_HOOKS.clear()
    print("PASS proxy_pin_switch")


if __name__ == "__main__":
    test_marlin_gate()
    test_proxy_pin_switch()
    print("2/2 PASS")
