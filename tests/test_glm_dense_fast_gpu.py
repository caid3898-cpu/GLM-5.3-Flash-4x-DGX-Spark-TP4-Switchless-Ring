"""GPU test for overlay/glm_dense_fast.py on ONE GB10 Spark (fleet stopped), inside the serving image, real weights.

  docker run --rm --gpus all --ipc=host --network none --memory 48g --memory-swap 48g \\
    -v $PWD:/w -v $MODEL_DIR:/model:ro -e GLM_DENSE_FAST_BUILD_DIR=/w/cache/build \\
    -e TRITON_CACHE_DIR=/w/cache/triton -e TORCH_CUDA_ARCH_LIST=12.1a -e PYTHONDONTWRITEBYTECODE=1 \\
    --entrypoint python3 glm53-roce:rel0928 /w/tests/test_glm_dense_fast_gpu.py --model /model --out /w/runs/dense_fast.jsonl
  options: --quick (parity only, no timing), --rounds N, --layer-kda 0 --layer-mla 3

What it does
  * loads only the tensors of one KDA layer (q/k/v/b/f_a/g_a/o, MXFP8) and one MLA layer (q_a/kv_a/q_b/o and the
    shared expert, block FP8) from the lossless8 checkpoint, cuts the TP4 rank-0 shard of each projection
    (in_proj_qkvbfg_a = q|k|v|b rows of rank 0 + the replicated f_a|g_a; o_proj = its K slice; ...), and turns them
    into served Marlin buffers with vLLM's own prepare functions (prepare_mxfp8_layer_for_marlin /
    process_fp8_weight_block_strategy + prepare_fp8_layer_for_marlin, the path MarlinMxfp8LinearKernel /
    MarlinFP8ScaledMMLinearKernel run at load);
  * installs the overlay on the image's real kernel classes and runs its load-time prepare (JIT build into
    GLM_DENSE_FAST_BUILD_DIR, self-test, tagging) exactly as the loader hook does;
  * parity, every shape, M = 1..32, 33, 64 and the prefill Ms (1024 .. 6912): overlay dispatch (apply_weights of
    the patched class) vs the stock apply_weights (Marlin) vs an fp64 reference built from the CHECKPOINT tensors
    (independent of the Marlin layout: MX w * 2^(e-127) exact; block bf16(w * bf16(s)), Marlin's operand). Pass:
    finite, close_to(fast, stock) (rel L2 <= 1e-3, |diff| <= 2 ulp + 1e-3 rms), fast-vs-ref <= max(2 x
    stock-vs-ref, 1e-3); a call the table routes to stock must be bit-identical to stock;
  * CUDA graphs: fast path captured on a side stream at M = 1, 4, 8, 16, 32; replay == eager bit for bit, also
    after the static input is overwritten; prefill-sized calls inside a capture take the stock path;
  * timing (decode): CUDA graphs of G calls over rotating weight copies (>= 96 MB, cold in the 24 MB L2), arms
    interleaved ABBA, median us/call per shape and M; (prefill) eager, events, alternating arms, median;
  * summary: ms/step saved at c1 (M = 4 .. 8) and c4 (M = 16 .. 32) with the per-step call counts, and ms saved per
    6912-token (and 2048-token) prefill chunk.
Exit 1 on any parity / graph failure.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))

FAILS = []
PFX = "model.language_model.layers."


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------------------------
# weights
# ------------------------------------------------------------------------------------------------------------------
class Index:
    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]

    def get(self, key):
        from safetensors import safe_open
        with safe_open(os.path.join(self.dir, self.map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def shards(idx, lk, lm, tp=4, rank=0):
    """{shape name: (fp8 weight [N, K], scale, mx)} for TP rank `rank`; scale = uint8 [N, K/32] (MX) or float32
    [N/128, K/128] (block)."""
    a = f"{PFX}{lk}.self_attn."
    m = f"{PFX}{lm}.self_attn."
    e = f"{PFX}{lm}.mlp.shared_experts."
    out = {}

    def rows(t, n):  # rank slice of the first dim, n rows per rank
        return t[rank * n:(rank + 1) * n]

    # KDA in_proj_qkvbfg_a: q, k, v (8192 each), b (64) sharded; f_a, g_a (128 each) replicated
    ws, ss = [], []
    for p, n in (("q_proj", 2048), ("k_proj", 2048), ("v_proj", 2048), ("b_proj", 16)):
        ws.append(rows(idx.get(a + p + ".weight"), n))
        ss.append(rows(idx.get(a + p + ".weight_scale"), n))
    for p in ("f_a_proj", "g_a_proj"):
        ws.append(idx.get(a + p + ".weight"))
        ss.append(idx.get(a + p + ".weight_scale"))
    out["kda_in_proj"] = (torch.cat(ws), torch.cat(ss), True)
    w, s = idx.get(a + "o_proj.weight"), idx.get(a + "o_proj.weight_scale")
    out["kda_o"] = (w[:, rank * 2048:(rank + 1) * 2048].contiguous(), s[:, rank * 64:(rank + 1) * 64].contiguous(), True)
    # MLA (block 128): fused q_a | kv_a replicated; q_b rows / o cols sharded
    out["mla_qkv_a"] = (torch.cat([idx.get(m + "q_a_proj.weight"), idx.get(m + "kv_a_proj_with_mqa.weight")]),
                        torch.cat([idx.get(m + "q_a_proj.weight_scale_inv"),
                                   idx.get(m + "kv_a_proj_with_mqa.weight_scale_inv")]), False)
    out["mla_q_b"] = (rows(idx.get(m + "q_b_proj.weight"), 4096), rows(idx.get(m + "q_b_proj.weight_scale_inv"), 32),
                      False)
    w, s = idx.get(m + "o_proj.weight"), idx.get(m + "o_proj.weight_scale_inv")
    out["mla_o"] = (w[:, rank * 4096:(rank + 1) * 4096].contiguous(), s[:, rank * 32:(rank + 1) * 32].contiguous(),
                    False)
    # shared expert: gate | up rows sharded (512 each per rank), down K sharded
    out["shared_gate_up"] = (torch.cat([rows(idx.get(e + "gate_proj.weight"), 512), rows(idx.get(e + "up_proj.weight"), 512)]),
                             torch.cat([rows(idx.get(e + "gate_proj.weight_scale_inv"), 4),
                                        rows(idx.get(e + "up_proj.weight_scale_inv"), 4)]), False)
    w, s = idx.get(e + "down_proj.weight"), idx.get(e + "down_proj.weight_scale_inv")
    out["shared_down"] = (w[:, rank * 512:(rank + 1) * 512].contiguous(), s[:, rank * 4:(rank + 1) * 4].contiguous(), False)
    return out


def marlin_layer(w, s, mx, dev):
    """A module holding the served Marlin buffers, prepared by vLLM's own functions."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mu
    N, K = w.shape
    lay = torch.nn.Module()
    lay.weight = torch.nn.Parameter(w.to(dev), requires_grad=False)
    lay.output_size_per_partition = N
    lay.input_size_per_partition = K
    lay.bias = None
    if mx:
        lay.weight_scale = torch.nn.Parameter(s.to(dev), requires_grad=False)
        mu.prepare_mxfp8_layer_for_marlin(lay)
    else:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import process_fp8_weight_block_strategy
        from vllm.model_executor.utils import replace_parameter
        lay.weight_scale_inv = torch.nn.Parameter(s.to(dev), requires_grad=False)
        lay.weight_block_size = [128, 128]
        lay.orig_dtype = torch.bfloat16
        lay.logical_widths = [N]
        ww, ss = process_fp8_weight_block_strategy(lay.weight, lay.weight_scale_inv)
        replace_parameter(lay, "weight", ww.data)
        replace_parameter(lay, "weight_scale_inv", ss.data)
        mu.prepare_fp8_layer_for_marlin(lay, size_k_first=False)
    return lay


def reference_weight(w, s, mx):
    """fp64 [K, N] of the operand Marlin multiplies, from the checkpoint tensors."""
    wf = w.float()
    if mx:
        sc = torch.exp2(s.to(torch.float32) - 127).repeat_interleave(32, 1)
        op = (wf * sc).to(torch.bfloat16)                                     # exact (power of two)
    else:
        sb = s.to(torch.bfloat16).float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        op = (wf * sb).to(torch.bfloat16)                                     # Marlin: bf16(e4m3 * bf16(s))
    return op.double().t().contiguous()


def ulp_stats(a, b):
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    big = torch.maximum(a.float().abs(), b.float().abs()) > 1e-2 * b.float().pow(2).mean().sqrt()
    d = (ai - bi).abs()
    return dict(maxulp_big=int(d[big].max()) if bool(big.any()) else 0,
                eq=round(float((a.view(torch.int16) == b.view(torch.int16)).float().mean()), 6))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------
# timing
# ------------------------------------------------------------------------------------------------------------------
def time_graphs(fns, G, rounds, stream):
    graphs = {}
    for n, f in fns.items():
        with torch.cuda.stream(stream):
            for i in range(G):
                f(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for i in range(G):
                f(i)
        g.replay()
        torch.cuda.synchronize()
        graphs[n] = g
    res = {n: [] for n in fns}
    names = list(fns)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order + order[::-1]:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            b.synchronize()
            res[n].append(a.elapsed_time(b) * 1000.0 / G)
    del graphs
    return {n: round(st.median(v), 3) for n, v in res.items()}


def time_eager(fns, reps):
    res = {n: [] for n in fns}
    names = list(fns)
    for n in names:
        fns[n]()
    torch.cuda.synchronize()
    for r in range(reps):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fns[n]()
            b.record()
            b.synchronize()
            res[n].append(a.elapsed_time(b) * 1000.0)
    return {n: round(st.median(v), 1) for n, v in res.items()}


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--layer-kda", type=int, default=0)
    ap.add_argument("--layer-mla", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="dense_fast_gpu.jsonl")
    args = ap.parse_args()
    os.environ.setdefault("GLM_DENSE_FAST", "1")
    os.environ.setdefault("GLM_DENSE_FAST_PREFILL", "1")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    fout = open(args.out, "w")

    def emit(row):
        fout.write(json.dumps(row) + "\n")
        fout.flush()

    print(f"GPU {torch.cuda.get_device_name(dev)} torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
    import glm_dense_fast as gdf
    from vllm.model_executor.kernels.linear.mxfp8 import marlin as mx_mod
    from vllm.model_executor.kernels.linear.scaled_mm import marlin as fp8_mod
    stock_mx = mx_mod.MarlinMxfp8LinearKernel.apply_weights
    stock_fp8 = fp8_mod.MarlinFP8ScaledMMLinearKernel.apply_weights
    gdf.install_mxfp8(mx_mod)
    gdf.install_fp8(fp8_mod)
    self_mx = object.__new__(mx_mod.MarlinMxfp8LinearKernel)
    self_fp8 = types.SimpleNamespace(block_quant=True, marlin_input_dtype=None)
    fast_mx = mx_mod.MarlinMxfp8LinearKernel.apply_weights
    fast_fp8 = fp8_mod.MarlinFP8ScaledMMLinearKernel.apply_weights

    t0 = time.time()
    idx = Index(args.model)
    raw = shards(idx, args.layer_kda, args.layer_mla)
    layers, refs = {}, {}
    for name, (w, s, mx) in raw.items():
        layers[name] = marlin_layer(w, s, mx, dev)
        refs[name] = reference_weight(w.to(dev), s.to(dev), mx)
    del raw
    torch.cuda.synchronize()
    print(f"weights: {time.time() - t0:.1f} s; " + ", ".join(
        f"{n} K={l.input_size_per_partition} N={l.output_size_per_partition} w{tuple(l.weight.shape)}"
        for n, l in layers.items()), flush=True)

    def calls(name):
        lay = layers[name]
        mx = gdf.SHAPES[name][2]
        s_self = self_mx if mx else self_fp8
        return (lambda x, l=lay: (fast_mx if mx else fast_fp8)(s_self, l, x)), \
               (lambda x, l=lay: (stock_mx if mx else stock_fp8)(s_self, l, x))

    # ---- load-time prepare (build + self-test + tagging), as the loader hook runs it ----
    model = types.SimpleNamespace(named_modules=lambda: [(f"L.{n}", l) for n, l in layers.items()])
    t0 = time.time()
    rep = gdf.prepare(gdf.scan(model, gdf.table()))
    t_prep = time.time() - t0
    check("prepare: every shape tagged, nothing off", rep["off"] == [] and len(rep["layers"]) == 7,
          f"{t_prep:.1f} s {rep['layers']} off={rep['off']}")
    emit({"kind": "prepare", "seconds": round(t_prep, 1), "report": {k: v for k, v in rep.items() if k != "tags"},
          "tags": rep["tags"], "scratch_mb": {k: v.numel() * 2 / 2**20 for k, v in gdf._STATE["scratch"].items()},
          "ws_bytes": {k: [v[0].numel() * 4, v[1].numel() * 4] for k, v in gdf._STATE["ws"].items()}})

    # ---- parity ----
    g = torch.Generator(device="cpu").manual_seed(7)
    prefill_Ms = [1024, 1536, 2048, 4096, 6912]
    for name, lay in layers.items():
        fast, stock = calls(name)
        tag = lay.__dict__.get(gdf.TAG) or gdf.Tag(name, 0, 0, 0, False, "")   # untagged: all stock
        K = lay.input_size_per_partition
        Ms = list(range(1, 33)) + [33, 64] + (prefill_Ms if name in ("kda_in_proj", "kda_o") else [2048])
        worst = {"rel_fast": 0.0, "rel_stock_ref": 0.0}
        for M in Ms:
            x = (torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16).to(dev)
            gdf.S["fast"].clear()
            y = fast(x)
            used_fast = bool(gdf.S["fast"])
            ys = stock(x)
            expect_fast = (1 <= M <= tag.dec_max_m and gdf.bucket(M) in tag.dec) or M >= tag.pre_min_m
            row = {"kind": "parity", "shape": name, "M": M, "path": "fast" if used_fast else "stock"}
            ok = used_fast == expect_fast and y.shape == ys.shape and bool(torch.isfinite(y.float()).all())
            if used_fast:
                r = gdf.close_to(y, ys)
                ref = (x.double() @ refs[name]).float()
                rf, rs = rel(y, ref), rel(ys, ref)
                row.update(close=r, rel_vs_stock=r["rel"], rel_fast_ref=rf, rel_stock_ref=rs, **ulp_stats(y, ys))
                ok = ok and r["ok"] and rf <= max(2 * rs, 1e-3)
                worst["rel_fast"] = max(worst["rel_fast"], r["rel"])
                worst["rel_stock_ref"] = max(worst["rel_stock_ref"], rs)
            else:
                row["bit_equal_stock"] = bool(torch.equal(y, ys))
                ok = ok and row["bit_equal_stock"]
            row["ok"] = ok
            emit(row)
            if not ok:
                check(f"parity {name} M={M}", False, json.dumps(row))
        fastMs = [M for M in Ms if (1 <= M <= tag.dec_max_m and gdf.bucket(M) in tag.dec) or M >= tag.pre_min_m]
        check(f"parity {name}", not any(f.startswith(f"parity {name} ") for f in FAILS),
              f"fast at M={fastMs[:3]}..{fastMs[-1]} ({len(fastMs)}), stock elsewhere; worst rel vs stock "
              f"{worst['rel_fast']:.2e} (stock vs fp64 ref up to {worst['rel_stock_ref']:.2e})")

    # ---- CUDA graphs ----
    side = torch.cuda.Stream()
    for name, lay in layers.items():
        fast, _ = calls(name)
        K = lay.input_size_per_partition
        tag = lay.__dict__.get(gdf.TAG) or gdf.Tag(name, 0, 0, 0, False, "")
        for M in (1, 4, 8, 16, 32):
            if not (M <= tag.dec_max_m and gdf.bucket(M) in tag.dec):
                continue
            xs = (torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16).to(dev)
            with torch.cuda.stream(side):
                fast(xs)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=side):
                yg = fast(xs)
            ok = True
            for trial in range(2):
                if trial:
                    xs.copy_((torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16))
                gr.replay()
                torch.cuda.synchronize()
                ok = ok and torch.equal(yg, fast(xs.clone()))
            if not ok:
                check(f"graph {name} M={M}", False, "replay != eager")
        check(f"graph {name}", not any(f.startswith(f"graph {name} ") for f in FAILS), "replay == eager, bitwise")
    if "kda_in_proj" in layers:
        fast, stock = calls("kda_in_proj")
        xs = (torch.randn((2048, 4096), generator=g) * 0.5).to(torch.bfloat16).to(dev)
        gdf.S["fast"].clear()
        with torch.cuda.stream(side):
            stock(xs)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=side):
            fast(xs)
        check("graph: prefill-sized call inside a capture takes the stock path", not gdf.S["fast"], str(gdf.S["fast"]))
        del gr

    if args.quick:
        return finish(fout)

    # ---- timing: decode (graphs, cold weights) ----
    summary = {}
    for name, lay in layers.items():
        tag = lay.__dict__.get(gdf.TAG) or gdf.Tag(name, 0, 0, 0, False, "")
        mx = gdf.SHAPES[name][2]
        K, N = lay.input_size_per_partition, lay.output_size_per_partition
        wb = lay.weight.numel() * 4 + (lay.weight_scale if mx else lay.weight_scale_inv).numel() * (1 if mx else 2)
        copies = max(2, -(-96 * 2**20 // wb))
        clones = []
        for _ in range(copies):
            c = torch.nn.Module()
            c.weight = torch.nn.Parameter(lay.weight.detach().clone(), requires_grad=False)
            sattr = "weight_scale" if mx else "weight_scale_inv"
            setattr(c, sattr, torch.nn.Parameter(getattr(lay, sattr).detach().clone(), requires_grad=False))
            c.workspace = lay.workspace
            c.output_size_per_partition, c.input_size_per_partition, c.bias = N, K, None
            c.__dict__[gdf.TAG] = tag
            clones.append(c)
        G = 2 * copies
        s_self = self_mx if mx else self_fp8
        fa, sa = (fast_mx, stock_mx) if mx else (fast_fp8, stock_fp8)
        for M in (1, 2, 4, 8, 12, 16, 24, 32):
            xs = [(torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16).to(dev) for _ in range(copies)]
            res = time_graphs({"stock": lambda i: sa(s_self, clones[i % copies], xs[i % copies]),
                               "fast": lambda i: fa(s_self, clones[i % copies], xs[i % copies])},
                              G, args.rounds, side)
            fast_used = M <= tag.dec_max_m and gdf.bucket(M) in tag.dec
            row = {"kind": "time_decode", "shape": name, "M": M, "stock_us": res["stock"], "fast_us": res["fast"],
                   "path": "fast" if fast_used else "stock", "speedup": round(res["stock"] / res["fast"], 3),
                   "GBs_fast": round(wb / res["fast"] / 1e3, 1), "calls_per_step": gdf.SHAPES[name][3]}
            emit(row)
            summary[(name, M)] = row
            print(f"  {name:15s} M={M:3d} stock {res['stock']:8.2f} us  fast {res['fast']:8.2f} us  "
                  f"x{row['speedup']:.3f}  ({row['path']})", flush=True)
        del clones, xs
        torch.cuda.empty_cache()

    # ---- timing: prefill (eager) ----
    pre = {}
    for name in ("kda_in_proj", "kda_o"):
        fast, stock = calls(name)
        K = layers[name].input_size_per_partition
        for M in prefill_Ms:
            x = (torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16).to(dev)
            res = time_eager({"stock": lambda: stock(x), "fast": lambda: fast(x)}, 9)
            tag = layers[name].__dict__.get(gdf.TAG) or gdf.Tag(name, 0, 0, 0, False, "")
            row = {"kind": "time_prefill", "shape": name, "M": M, "stock_us": res["stock"], "fast_us": res["fast"],
                   "path": "fast" if M >= tag.pre_min_m else "stock", "speedup": round(res["stock"] / res["fast"], 3),
                   "calls_per_chunk": gdf.SHAPES[name][3]}
            emit(row)
            pre[(name, M)] = row
            print(f"  {name:15s} M={M:5d} stock {res['stock']:9.1f} us  fast {res['fast']:9.1f} us  "
                  f"x{row['speedup']:.3f}  ({row['path']})", flush=True)

    # ---- summary ----
    def step_saved(M):
        return round(sum(r["calls_per_step"] * (r["stock_us"] - r["fast_us"]) for (n, m), r in summary.items()
                         if m == M and r["path"] == "fast") / 1000, 3)

    tot = {f"M={M}": step_saved(M) for M in (1, 2, 4, 8, 12, 16, 24, 32)}
    chunk = {f"M={M}": round(sum(r["calls_per_chunk"] * (r["stock_us"] - r["fast_us"]) for (n, m), r in pre.items()
                                 if m == M and r["path"] == "fast") / 1000, 2) for M in prefill_Ms}
    emit({"kind": "summary", "ms_per_step_saved": tot, "ms_per_prefill_chunk_saved": chunk})
    print(f"ms/step saved (decode, all covered layers): {tot}", flush=True)
    print(f"ms saved per prefill chunk (34 KDA layers): {chunk}", flush=True)
    return finish(fout)


def finish(fout):
    fout.close()
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
