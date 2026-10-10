# SPDX-License-Identifier: Apache-2.0
"""CPU placement for the vLLM processes on GB10 (10 Cortex-X925 + 10 Cortex-A725 cores). Default OFF.

GLM_DS_CPU_PIN=1 pins, by vLLM process title (set_process_title), every thread of:
  VLLM::EngineCore*  -> GLM_DS_CPU_PIN_ENGINE  (default "big")
  VLLM::Worker*      -> GLM_DS_CPU_PIN_WORKER  (default "big")
  the API server     -> GLM_DS_CPU_PIN_API     (default "all": unpinned)
Values: "big" (X925), "little" (A725), "all", or an explicit list "5-9,15-19".
Big cores are detected from /sys/.../cpu_capacity (highest capacity), else from MIDR part 0xd85 (X925),
else GLM_DS_CPU_BIG (default "5-9,15-19", the numbering given for our nodes).

DS4.1 measured this flat on SGLang (live taskset ABAB, 2026-09-25: CUDA graphs replayed and the host ran
~35 ms ahead). vLLM keeps more of the step eager on the host (sampler, lm_head, draft prep, DFlash context
K/V precompute) and does scheduling + ZMQ fan-out to three remote workers per step, so this is re-tested
here; scripts/cpu_pin_live.sh does the same thing live (no reboot) for an in-boot ABAB, which is the
preferred first test. This module is the persistent form.
"""
from __future__ import annotations

import os
import sys

_DEFAULT_BIG = "5-9,15-19"


def parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def fmt_cpus(cpus) -> str:
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def detect_big(sysfs: str = "/sys/devices/system/cpu", online: set[int] | None = None) -> tuple[set[int], str]:
    """Return (big core set, how it was found)."""
    online = online if online is not None else set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    caps = {}
    for c in sorted(online):
        try:
            with open(f"{sysfs}/cpu{c}/cpu_capacity") as f:
                caps[c] = int(f.read().strip())
        except (OSError, ValueError):
            pass
    if caps and len(set(caps.values())) > 1:
        top = max(caps.values())
        return {c for c, v in caps.items() if v == top}, "cpu_capacity"
    parts = {}
    for c in sorted(online):
        try:
            with open(f"{sysfs}/cpu{c}/regs/identification/midr_el1") as f:
                parts[c] = (int(f.read().strip(), 16) >> 4) & 0xFFF
        except (OSError, ValueError):
            pass
    if parts and 0xD85 in parts.values():
        return {c for c, p in parts.items() if p == 0xD85}, "midr X925 (0xd85)"
    return parse_cpus(os.environ.get("GLM_DS_CPU_BIG", _DEFAULT_BIG)), "GLM_DS_CPU_BIG/default"


def resolve(spec: str, big: set[int], online: set[int]) -> set[int] | None:
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "none", "off"):
        return None
    if spec == "big":
        return big & online or None
    if spec == "little":
        return (online - big) or None
    cpus = parse_cpus(spec) & online
    return cpus or None


def role_of(title: str) -> str | None:
    t = title.split("::", 1)[-1]
    if t.startswith("EngineCore"):
        return "engine"
    if t.startswith("Worker"):
        return "worker"
    if t.startswith("APIServer"):
        return "api"
    return None


def pin_all_threads(cpus: set[int], pid: int | None = None) -> int:
    """sched_setaffinity on every task of the process (threads created later inherit from their creator)."""
    pid = os.getpid() if pid is None else pid
    n = 0
    try:
        tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        tids = [0]
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpus)
            n += 1
        except OSError:
            pass
    return n


def _targets():
    online = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    try:  # the container's full set, not an inherited narrower one
        with open("/sys/devices/system/cpu/online") as f:
            online = parse_cpus(f.read().strip()) or online
    except OSError:
        pass
    big, how = detect_big(online=online)
    return {
        "engine": resolve(os.environ.get("GLM_DS_CPU_PIN_ENGINE", "big"), big, online),
        "worker": resolve(os.environ.get("GLM_DS_CPU_PIN_WORKER", "big"), big, online),
        "api": resolve(os.environ.get("GLM_DS_CPU_PIN_API", "all"), big, online),
        "_all": online, "_big": big, "_how": how,
    }


def apply_role(role: str, title: str = "") -> None:
    t = _targets()
    cpus = t.get(role) or t["_all"]
    if not cpus:
        return
    n = pin_all_threads(cpus)
    print(f"glm-ds cpu-pin: {title or role} pid {os.getpid()} -> cpus {fmt_cpus(cpus)} ({n} threads; "
          f"big={fmt_cpus(t['_big'])} via {t['_how']})", file=sys.stderr, flush=True)


def pin_main_process() -> None:
    """The API server is the container's main `vllm serve` process; spawned children re-pin by title."""
    argv = " ".join(sys.argv)
    if "multiprocessing" in argv or "spawn_main" in argv:
        return
    if os.path.basename(sys.argv[0] if sys.argv else "") == "vllm" or " serve " in f" {argv} ":
        apply_role("api", "APIServer(main)")


def install(system_utils_module) -> None:
    mod = system_utils_module
    if getattr(mod, "_glm_ds_cpu_pin", False):
        return
    orig = mod.set_process_title

    def set_process_title(name, suffix="", prefix=None):
        if prefix is None:
            orig(name, suffix)
        else:
            orig(name, suffix, prefix)
        full = f"{name}_{suffix}" if suffix else name
        role = role_of(full)
        if role is not None:
            try:
                apply_role(role, full)
            except Exception as exc:  # noqa: BLE001 - placement must never kill a process
                print(f"glm-ds cpu-pin: {full}: {exc!r}", file=sys.stderr, flush=True)

    mod.set_process_title = set_process_title
    mod._glm_ds_cpu_pin = True
