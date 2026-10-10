"""CPU check for overlay/glm_prefill_hooks.py: the after-import hook, register() gating, and the
drift-guard tables of glm_prefill_shard / glm_prefill_sched against an extracted engine tree.

  python3 tests/test_glm_prefill_hooks.py [--tree ROOT]         run the checks
  python3 tests/test_glm_prefill_hooks.py --print-hashes ROOT     print the tables for ROOT

ROOT is a directory holding `vllm/...` exactly as installed in the image (for the pinned v11 image:
vLLM 487ecf187 overlaid with the files extracted from the image). Without --tree / GLM_VLLM_TREE the
table check is skipped. On a node, the same comparison runs against the live image via
`docker run --rm --entrypoint python3 $IMAGE - < print_image_hashes.py` (RESULTS.md, fleet step 0).
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))

import glm_prefill_hooks as hooks  # noqa: E402


def tables():
    import ast
    out = {}
    for name in ("glm_prefill_shard.py", "glm_prefill_sched.py"):
        src = open(os.path.join(HERE, "..", "overlay", name)).read()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EXPECTED" for t in node.targets):
                out[name] = ast.literal_eval(node.value)
    return out


def hashes_for(root: str, expected: dict) -> dict:
    got = {}
    for key in expected:
        mod, qual = key.split(":", 1)
        path = os.path.join(root, *mod.split(".")) + ".py"
        try:
            got[key] = hooks.src_hash(path, qual)
        except (OSError, KeyError) as exc:
            got[key] = f"missing ({type(exc).__name__})"
    return got


def test_after_import_runs_after_exec():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "glmhookpkg")
        os.makedirs(pkg)
        open(os.path.join(pkg, "__init__.py"), "w").close()
        with open(os.path.join(pkg, "target.py"), "w") as f:
            f.write("VALUE = 1\n")
        sys.path.insert(0, d)
        seen = []
        try:
            hooks.after_import("glmhookpkg.target", lambda m: seen.append(m.VALUE))
            import glmhookpkg.target as t  # noqa: F401
            assert seen == [1], seen
            hooks.after_import("glmhookpkg.target", lambda m: seen.append("again"))   # already imported
            assert seen == [1, "again"], seen
        finally:
            sys.path.remove(d)


def test_register_is_inert_when_off():
    assert hooks.register({}) == {"shard": False, "sched": False}
    assert hooks.register({"GLM_PREFILL_SHARD": "0", "GLM_PREFILL_CADENCE": "0"}) == {"shard": False, "sched": False}
    assert hooks.wanted({"GLM_PREFILL_SHARD": "comm"})["shard"]
    assert hooks.wanted({"GLM_IDLE_COALESCE_MS": "4"})["sched"]


def test_func_source_is_stable():
    src = textwrap.dedent('''
        class A:
            @staticmethod
            def f(x):
                return x  # c

        def g():
            pass
    ''')
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        assert hooks.func_source(f.name, "A.f") == "def f(x):\n    return x  # c\n"
        assert hooks.func_source(f.name, "g") == "def g():\n    pass\n"
    finally:
        os.unlink(f.name)


def check_tree(root: str) -> None:
    bad = []
    for name, exp in tables().items():
        got = hashes_for(root, exp)
        bad += [(name, k, exp[k], got[k]) for k in exp if got[k] != exp[k]]
    for row in bad:
        print("MISMATCH", *row)
    assert not bad, f"{len(bad)} source hashes differ from {root}"


def main(argv):
    if len(argv) >= 2 and argv[0] == "--print-hashes":
        for name, exp in tables().items():
            for k, v in hashes_for(argv[1], exp).items():
                print(f"{name}  {k}  {v}{'' if v == exp[k] else '   (table: ' + exp[k] + ')'}")
        return
    tree = argv[1] if len(argv) >= 2 and argv[0] == "--tree" else os.environ.get("GLM_VLLM_TREE")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    if tree:
        check_tree(tree)
        print("PASS drift tables match", tree)
    else:
        print("SKIP drift tables (no --tree / GLM_VLLM_TREE)")


if __name__ == "__main__":
    main(sys.argv[1:])
