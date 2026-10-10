"""8-bit non-expert weights for nvidia/GLM-5.3-Flash-NVFP4 (ModelOpt): MXFP8, calibration-free, streaming.

Why: nvidia's pack quantises the routed experts to NVFP4 but leaves attention (KDA q/k/v/b/f_a/g_a/o, MLA
q_b/kv_b/o) and the shared experts in BF16. Those ~14 GiB are read on every decode step (3.5 GiB per rank at
TP4). MXFP8 halves them at about 2-3 % relative weight error (NVFP4 would quarter them at about 9 %).

Why MXFP8 and not block-128 FP8: this is a ModelOpt checkpoint, and vLLM's ModelOpt MIXED_PRECISION loader
offers per-layer FP8 only as per-tensor W8A8 with a *calibrated static activation scale*, while MXFP8 is
weight-only on GB10 (Marlin W8A16, activations stay BF16) and needs no calibration. The on-disk format is the
one vLLM's ModelOptMxFp8LinearMethod loads: `weight` float8_e4m3fn [N, K] and `weight_scale` uint8 [N, K/32]
holding an E8M0 exponent (value = 2**(s - 127)).

Module set: the one tonyd2wild proved boots on this architecture with NVFP4 attention
(runs/2026-09-20-nvidia-lanes in github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark): the fused KDA
in_proj_qkvbfg_a (q, k, v, b, f_a, g_a), every o_proj, MLA q_b_proj / kv_b_proj, and the shared experts. Left in
BF16: the DSA indexer, f_b/g_b gates, q_a/kv_a, the router, embeddings, lm_head, the vision tower. It needs his
two-line glm5next patch (overlay/glm5next_kda.py, overlay/glm5next_model.py) because the image builds attention
with quant_config=None. The MTP layer (45) is dropped: DFlash2 does the drafting and the main model skips it.

    python3 quantize_nonexpert_mxfp8.py --self-test
    python3 quantize_nonexpert_mxfp8.py --src NVIDIA_DIR --dst OUT_DIR [--shards 1-21] --write
    python3 quantize_nonexpert_mxfp8.py --src NVIDIA_DIR --dst OUT_DIR --finalize   # config + index

Memory: one tensor at a time (largest 128 MB BF16); unchanged tensors are copied as raw bytes.
"""
import argparse, json, math, os, re, struct, sys, time

import torch

FP8_MAX = 448.0
BLOCK = 32
MTP_LAYER = 45

ATTN = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\."
                  r"(q_proj|k_proj|v_proj|b_proj|f_a_proj|g_a_proj|o_proj|q_b_proj|kv_b_proj)\.weight$")
SHARED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$")
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

# Proven ignore list (tonyd2wild tools/fix_ignore.py): everything that must stay BF16 under this module set.
IGNORE = ["lm_head", "model.language_model.embed_tokens",
          "*.self_attn.f_b_proj", "*.self_attn.g_b_proj", "*.self_attn.fused_qkv_a_proj",
          "*.self_attn.q_a_proj", "*.self_attn.kv_a_proj_with_mqa",
          "*.self_attn.indexer.wk_weights_proj", "*.self_attn.indexer.wk",
          "*.self_attn.indexer.weights_proj", "*.self_attn.indexer.wq_b",
          "*.mlp.gate", "*.eh_proj", "model.visual.*"]

DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "I64": 8, "I32": 4, "BOOL": 1}


def mxfp8(w: torch.Tensor):
    """BF16 [N, K] -> (float8_e4m3fn [N, K], uint8 E8M0 [N, K/32]). Exponent = ceil(log2(amax/448)), so no
    value is clipped; dequant = q * 2**(s-127)."""
    n, k = w.shape
    assert k % BLOCK == 0, f"K={k} not a multiple of {BLOCK}"
    b = w.to(torch.float32).view(n, k // BLOCK, BLOCK)
    amax = b.abs().amax(-1).clamp(min=2.0 ** -126)
    e = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    q = (b / torch.exp2(e).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(n, k).contiguous(), (e + 127).to(torch.uint8).contiguous()


def de_mxfp8(q, s):
    n, k = q.shape
    return (q.to(torch.float32).view(n, k // BLOCK, BLOCK) * torch.exp2(s.to(torch.float32) - 127).unsqueeze(-1)).view(n, k)


def self_test():
    torch.manual_seed(0)
    for shape in [(8192, 4096), (4096, 16384), (16, 4096), (2048, 128), (512, 4096)]:
        w = (torch.randn(shape) * 0.02).to(torch.bfloat16)
        q, s = mxfp8(w)
        rel = (de_mxfp8(q, s) - w.float()).norm() / w.float().norm()
        print(f"  {str(shape):14} mxfp8 rel-rms {rel:.4f}")
        assert rel < 0.04
    print("  self-test OK")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


def target(name):
    return ATTN.match(name) or SHARED.match(name)


def convert_shard(src, dst):
    hdr, base, meta = read_header(src)
    out, order = {}, []
    changed = False
    for name, t in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        m = LAYER.match(name)
        if m and int(m.group(1)) == MTP_LAYER:
            changed = True
            continue
        if target(name) and t["dtype"] == "BF16" and len(t["shape"]) == 2 and t["shape"][1] % BLOCK == 0:
            n, k = t["shape"]
            order.append(("q", name, t, n, k))
            changed = True
        else:
            order.append(("raw", name, t, None, None))
    if not changed:
        return None
    # layout: compute sizes first, so the header can be written before the data
    entries, off = [], 0
    for kind, name, t, n, k in order:
        if kind == "raw":
            sz = t["data_offsets"][1] - t["data_offsets"][0]
            entries.append((name, {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + sz]}, kind, t))
            off += sz
        else:
            wname = name
            sname = name[: -len(".weight")] + ".weight_scale"
            entries.append((wname, {"dtype": "F8_E4M3", "shape": [n, k], "data_offsets": [off, off + n * k]}, "qw", t))
            off += n * k
            entries.append((sname, {"dtype": "U8", "shape": [n, k // BLOCK], "data_offsets": [off, off + n * k // BLOCK]}, "qs", t))
            off += n * k // BLOCK
    newhdr = {"__metadata__": meta or {"format": "pt"}}
    newhdr.update({name: info for name, info, _, _ in entries})
    hb = json.dumps(newhdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = dst + ".part"
    nq = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        pending_scale = None
        for name, info, kind, t in entries:
            a, b = t["data_offsets"]
            if kind == "raw":
                fi.seek(base + a)
                left = b - a
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    fo.write(chunk)
                    left -= len(chunk)
            elif kind == "qw":
                fi.seek(base + a)
                raw = bytearray(fi.read(b - a))
                w = torch.frombuffer(raw, dtype=torch.bfloat16).view(*t["shape"])
                q, s = mxfp8(w)
                fo.write(q.view(torch.uint8).numpy().tobytes())
                pending_scale = s
                nq += 1
            else:
                fo.write(pending_scale.numpy().tobytes())
                pending_scale = None
    os.replace(tmp, dst)
    return {name: info for name, info, _, _ in entries}, nq


def parse_range(r, n):
    if not r:
        return list(range(1, n + 1))
    a, b = r.split("-")
    return list(range(int(a), int(b) + 1))


def finalize(src, dst):
    cfg = json.load(open(os.path.join(src, "config.json")))
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = {}
    shards = sorted(set(idx["weight_map"].values()))
    for f in shards:
        p = os.path.join(dst, f)
        assert os.path.exists(p), f"missing {p}"
        h, _, _ = read_header(p)
        for k in h:
            wm[k] = f
    idx["weight_map"] = dict(sorted(wm.items()))
    idx.setdefault("metadata", {})["total_size"] = sum(os.path.getsize(os.path.join(dst, f)) for f in shards)
    json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    # quantized_layers: experts + dense MLP stay NVFP4 (weight-only Marlin, as tonyd2wild's W4A16 lanes);
    # attention + shared experts are MXFP8.
    ql = {}
    text = cfg.get("text_config", cfg)
    nl = text["num_hidden_layers"]
    kinds = text.get("mlp_layer_types") or []
    for L in range(nl):
        p = f"model.language_model.layers.{L}"
        if L < len(kinds) and kinds[L] == "dense":
            for x in ("gate_proj", "up_proj", "down_proj"):
                ql[f"{p}.mlp.{x}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        else:
            ql[f"{p}.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            for x in ("gate_proj", "up_proj", "down_proj"):
                ql[f"{p}.mlp.shared_experts.{x}"] = {"quant_algo": "MXFP8"}
    for k in wm:
        m = ATTN.match(k)
        if m and wm.get(k[: -len(".weight")] + ".weight_scale"):
            ql[k[: -len(".weight")]] = {"quant_algo": "MXFP8"}
            if m.group(2) in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"):
                ql[f"model.language_model.layers.{m.group(1)}.self_attn.in_proj_qkvbfg_a"] = {"quant_algo": "MXFP8"}
    q = cfg["quantization_config"]
    newq = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": 16,
            "kv_cache_scheme": q.get("kv_cache_scheme"), "ignore": IGNORE, "quantized_layers": dict(sorted(ql.items())),
            "producer": {"name": "modelopt", "note": "nvidia NVFP4 experts + MXFP8 non-expert (quantize_nonexpert_mxfp8.py)"}}
    cfg["quantization_config"] = newq
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    hq = {"producer": newq["producer"], "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
          "group_size": 16, "exclude_modules": IGNORE, "quantized_layers": newq["quantized_layers"]}}
    json.dump(hq, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=1)
    import shutil
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
            continue
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, f))
    n_mx = sum(1 for v in ql.values() if v["quant_algo"] == "MXFP8")
    print(f"finalize: {len(wm)} tensors, quantized_layers {len(ql)} ({n_mx} MXFP8), total {idx['metadata']['total_size']/2**30:.1f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--dst")
    ap.add_argument("--shards", default="", help="e.g. 1-21 (1-based, inclusive)")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.self_test:
        self_test()
    if a.write:
        os.makedirs(a.dst, exist_ok=True)
        files = sorted(f for f in os.listdir(a.src) if re.match(r"model-\d+-of-\d+\.safetensors$", f))
        total = int(re.search(r"of-(\d+)", files[0]).group(1)) if files else 0
        want = set(parse_range(a.shards, total))
        for f in files:
            i = int(re.match(r"model-(\d+)", f).group(1))
            if i not in want:
                continue
            s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
            if os.path.exists(d):
                print(f"  skip {f} (exists)"); continue
            t0 = time.time()
            r = convert_shard(s, d)
            if r is None:
                try:
                    os.link(s, d)
                except OSError:  # different mounts (e.g. docker binds): copy instead
                    __import__("shutil").copyfile(s, d)
                print(f"  {f}: unchanged, linked/copied", flush=True)
            else:
                print(f"  {f}: {r[1]} tensors -> MXFP8, {time.time()-t0:.0f} s", flush=True)
    if a.finalize:
        finalize(a.src, a.dst)


if __name__ == "__main__":
    main()
