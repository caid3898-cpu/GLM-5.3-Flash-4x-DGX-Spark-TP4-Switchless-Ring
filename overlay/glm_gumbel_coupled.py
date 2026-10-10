# SPDX-License-Identifier: Apache-2.0
"""GLM_GUMBEL_COUPLED=1|check: Gumbel-coupled DFlash2 drafting at temperature > 0. Exact. Default OFF.

Port of llama.cpp-lab PR #26, "speculative : coupled sampling for a sampling target (--spec-coupled)", by Jim Routh
(routhjim, https://github.com/routhjim/llama.cpp-lab/pull/26, merged 2026-09-07), to the vLLM V2 model runner of
the Tony v11 image (vLLM 0.1.dev20051+g487ecf187, DFlash2 speculator transplanted from upstream b389ac294).
Credit for the method and its measurements is theirs; the vLLM port, kernels and checks are ours. Theory:
Gumbel-max trick (Gumbel 1954; Maddison, Tarlow and Minka 2014); drafter-invariant coupling (Daliri et al.,
arXiv 2408.07978).

The noise. The V2 sampler already draws Gumbel noise as a pure function of (request seed, position, token id):
    gumbel.py gumbel_block_argmax:  u = tl.rand(tl.randint(seed, pos), token_id);  g = -log(-log1p(-u))
Philox is counter-based, so g(seed, pos, v) is the same value wherever it is computed. The target's row whose input
sits at position P (it predicts the token at P + 1) uses key pos = P, in the plain Sampler and in the rejection
sampler alike. The image's DFlash2 walk (dflash2/speculator.py _selector_walk_kernel -> gumbel_noised_argmax) keys
draft token j (placed at position sample_pos) with pos = sample_pos - 1: the SAME key as the target row that
verifies it. Production drafts greedily (draft_sample_method unset), so that noise path is never taken today.

What this overlay changes, only while GLM_GUMBEL_COUPLED is on:
  draft   DFlash2Speculator._sample_path runs the walk WITH the keyed noise and without draft logits:
            d_j = argmax_{c in C_j} ( s_j(c | d_{j-1}) / (tau T) + g(seed, P_j, c) )
          restricted (GLM_GUMBEL_COUPLED_DRAFT_TOPP=1, default) to the request's top-p nucleus of softmax(s_j / T)
          over the 16 candidates, as in the PR. Greedy rows (T = 0) walk the plain argmax, exactly as stock.
  verify  RejectionSampler._verify: the stock apply_sampling_params (penalties, logit bias / min_tokens, bad words,
          thinking budget, temperature, min-p, top-k / top-p: the same calls, same order, same row count), then the
          STOCK gumbel_sample over every verify row (apply_temperature=False, as Sampler.sample does):
            t_i = argmax_v ( processed_i(v) + g(seed, P_i, v) )
          and accept-on-match (glm_target_argmax.greedy_verify, the stock greedy-branch rule): commit t_0 .. t_n,
          where n = the number of leading drafts equal to t. A draft that is -1 (verify cut / padding) never matches
          and simply ends the run. There is no residual distribution and no forced rejection: every committed token
          is the target's own draw t_i.

Why it is exact. Row i's committed token t_i depends only on the processed target logits of its prefix and on
g(seed, P_i, .); the prefix is made of earlier committed tokens, i.e. of logits and noise at earlier positions.
Noise at different positions is independent, so t_i ~ softmax(processed_i) given the prefix: the output process
is the target's sampling process. The draft never enters a committed token; it only decides how many positions
one step commits. Stronger: for identical logits the committed stream is bit-identical to the stock
non-speculative Sampler with the same seed (its gumbel_sample path: explicit seed or no top-k/p; without a seed and
with top-p the stock plain Sampler uses FlashInfer's sampler, another RNG with the same distribution). At T = 0
the gumbel kernel is a plain argmax (lowest id on ties, like the stock greedy verify), so greedy output is unchanged.
Numerics: target logits depend (bf16 GEMM shapes) on the verify width, which the draft sets; that is the same
batch-shape dependence the stock speculative path has, and it is why text is compared per step, not per request.

What would NOT be exact, and is refused: a noise-sampled draft verified by the STOCK rejection sampler. Its
one-hot residual (target minus the draft token) is resampled with the same g the draft was drawn with, i.e.
conditioned on the draft having won, which biases the residual. So draft mode and verify mode are one switch;
under the in-boot A/B the drafter graphs must be captured per variant (GLM_AB_DRAFT_SETS=1) whenever variants
differ on GLM_GUMBEL_COUPLED, else the boot is refused. Synthetic and block verification are refused too.

Modes (glm_ab key "mode": switchable per in-boot variant; draft side baked into each drafter graph set)
  GLM_GUMBEL_COUPLED=0|1|check
      check: also run the stock Sampler.sample (its gumbel path; FlashInfer forced off, same logits, same rows) and
      count per step: committed tokens != the stock draw at their position, accepted drafts != the stock draw,
      a rejected draft that equals the stock draw, num_sampled out of range. Serves the coupled result. Syncs.
  GLM_GUMBEL_COUPLED_TAU=1.0          draft scale (the PR: 1.0 is right, 0.3-0.6 collapse to argmax, >= 2 hurts)
  GLM_GUMBEL_COUPLED_DRAFT_TOPP=1     draft-side top-p over the 16 candidates (acceptance only)
  GLM_GUMBEL_COUPLED_LOG_EVERY=2000   stats line cadence (TP rank 0)
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import os
import sys
import textwrap

import numpy as np

_OFF = ("", "0", "off", "false", "no")
KEY = "GLM_GUMBEL_COUPLED"
_MODE = os.environ.get(KEY, "0").strip().lower()
if _MODE not in _OFF + ("1", "on", "true", "check"):
    raise ValueError(f"{KEY}={_MODE!r} is not 0|1|check")
TAU = float(os.environ.get("GLM_GUMBEL_COUPLED_TAU", "1.0") or 1.0)
DRAFT_TOPP = os.environ.get("GLM_GUMBEL_COUPLED_DRAFT_TOPP", "1").strip().lower() not in _OFF
LOG_EVERY = int(os.environ.get("GLM_GUMBEL_COUPLED_LOG_EVERY", "2000"))
if not 0.05 <= TAU <= 8.0:
    raise ValueError(f"GLM_GUMBEL_COUPLED_TAU={TAU} out of range [0.05, 8]")

RS = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
DF2 = "vllm.v1.worker.gpu.spec_decode.dflash2.speculator"
SAMPLER = "vllm.v1.worker.gpu.sample.sampler"

# function text sha256[:16] in the qualified image (glm53-roce:rel0928 = Tony v11 + our mounts; gumbel.py is the
# mounted overlay/gumbel.py with the vllm#50843 clamp). Same scheme as glm_exact_hooks.src_hash.
EXPECTED = {
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._verify": "e8e2dbd71fa67f53",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._verify_in_chunks": "4e1a2f1099e88bc4",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler.__call__": "5a297f779d7f774f",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler.__init__": "468fc771339b0472",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.apply_sampling_params": "56558b1afa204b45",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.sample": "0a9891bf03af6e71",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.__init__": "e2e5a620896502f7",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.add_request": "61b74ce385164c80",
    "vllm.v1.worker.gpu.sample.states:SamplingStates.add_request": "80021510b9c0a196",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_sample": "ded980830618e668",
    "vllm.v1.worker.gpu.sample.gumbel:_gumbel_sample_kernel": "afcc8d4e2757c889",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_block_argmax": "96bf9de0671d9e25",
    "vllm.v1.worker.gpu.sample.gumbel:tl_rand32": "13ea069c256c3051",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:gumbel_noised_argmax": "ff0a5ca464e56265",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:_selector_walk_kernel": "6cfef8e1b7fc8cbf",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._sample_path": "d67af3e6de394fa9",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._generate_draft": "2550bc0960238491",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.propose": "432c21dae8db3abe",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:_prepare_dflash_inputs_kernel": "f2929acf37e397b1",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample": "d58876dc84455870",
}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-gumbel-coupled: {msg}\n")


def _mode() -> str:
    """Per-call mode: the in-boot A/B harness (glm_ab, TEST ONLY) switches it per variant; otherwise the env."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and KEY in ab.KNOWN:
        return ab.norm_value(KEY, ab.env(KEY))
    if _MODE in _OFF:
        return "0"
    return "check" if _MODE == "check" else "1"


# ---------------------------------------------------------------------------------------------------
# Reference noise (numpy port of Triton's Philox4x32-10 randint / rand, triton/language/random.py of the image):
# used by the CPU tests, the GPU test's cross-check and nothing on the serving path.
# ---------------------------------------------------------------------------------------------------
_M32 = np.uint64(0xFFFFFFFF)


def _philox_c0(seed_u64: np.ndarray, off_lo: np.ndarray, off_hi: np.ndarray, rounds: int = 10) -> np.ndarray:
    """Triton philox(seed, c0=off_lo, c1=off_hi, 0, 0) with 32-bit counters -> c0 (uint64 holding a uint32)."""
    A, B = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    KA, KB = np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
    s = seed_u64.astype(np.uint64)
    k0 = s & _M32
    k1 = (s >> np.uint64(32)) & _M32
    c0 = off_lo.astype(np.uint64) & _M32
    c1 = off_hi.astype(np.uint64) & _M32
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0, k1 = np.broadcast_arrays(k0, k1)
    k0, k1 = k0.copy(), k1.copy()
    for _ in range(rounds):
        _c0, _c2 = c0, c2
        c0 = (((B * _c2) >> np.uint64(32)) ^ c1 ^ k0) & _M32
        c2 = (((A * _c0) >> np.uint64(32)) ^ c3 ^ k1) & _M32
        c1 = (B * _c2) & _M32
        c3 = (A * _c0) & _M32
        k0 = (k0 + KA) & _M32
        k1 = (k1 + KB) & _M32
    return c0


def _i64_as_u64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).view(np.uint64)


def gumbel_noise_np(seed: int, pos: int, tokens) -> np.ndarray:
    """g(seed, pos, token) of gumbel_block_argmax (fp32 path), float32. The logs are numpy's, so a result can differ
    from the GPU's in the last ulp; argmax comparisons in the tests allow for that at near-ties."""
    s = _i64_as_u64(np.array([seed], dtype=np.int64))
    p = _i64_as_u64(np.array([pos], dtype=np.int64))
    gseed = _philox_c0(s, p & _M32, (p >> np.uint64(32)) & _M32)            # tl.randint(seed, pos): uint32
    t = np.asarray(tokens, dtype=np.int64)
    tu = _i64_as_u64(t)
    r = _philox_c0(np.broadcast_to(gseed, tu.shape), tu & _M32, (tu >> np.uint64(32)) & _M32)
    x = r.astype(np.uint32).view(np.int32).astype(np.int64)                  # bitcast to int32
    x = np.where(x < 0, -x - 1, x)
    u = x.astype(np.float32) * np.float32(4.6566127342e-10)
    u = np.maximum(u, np.float32(4.6566127342e-10))
    return (-np.log(-np.log1p(-u.astype(np.float32)))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# pure torch / numpy pieces (CPU-testable)
# ---------------------------------------------------------------------------------------------------
def draft_topp_keep(scores, temperature: float, top_p: float):
    """Candidates kept by the draft-side nucleus: probability mass of STRICTLY larger scores < top_p (torch)."""
    import torch
    z = scores.float() / temperature
    p = torch.softmax(z, dim=-1)
    gt = z.unsqueeze(-1) > z.unsqueeze(-2)                                   # [..., i, j]: z_i > z_j
    above = (p.unsqueeze(-1) * gt).sum(dim=-2)
    return above < top_p


def walk_torch(scores, candidates, positions, temperature: float, seed: int, top_p: float = 1.0,
               tau: float = 1.0, noise=None):
    """Reference coupled walk for one request. scores [K steps, C prev, C cand] fp32, candidates [K, C] int64,
    positions [K] = sample_pos - 1 (the verify row's key). noise(seed, pos, tokens) -> tensor. Returns [K] tokens."""
    import torch
    out = []
    prev = 0
    for j in range(scores.shape[0]):
        s = scores[j, prev].float()
        if temperature == 0.0:
            idx = int(torch.argmax(s))
        else:
            sel = s * (1.0 / tau)
            if top_p < 1.0:
                sel = torch.where(draft_topp_keep(s, temperature, top_p), sel, torch.full_like(sel, -float("inf")))
            g = noise(seed, int(positions[j]), candidates[j])
            idx = int(torch.argmax(sel / temperature + g))
        out.append(int(candidates[j, idx]))
        prev = idx
    return out


# ---------------------------------------------------------------------------------------------------
# Triton kernel (built lazily; CUDA only)
# ---------------------------------------------------------------------------------------------------
_K: dict = {}


def _kernels():
    if _K:
        return _K
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import gumbel_noised_argmax

    @triton.jit
    def _coupled_walk_kernel(scores_ptr, candidate_ptr, sample_pos_ptr, req_state_ptr, temperature_ptr, seeds_ptr,
                             top_p_ptr, tokens_ptr, realized_scores_ptr, inv_tau,
                             num_steps: tl.constexpr, top_k: tl.constexpr, BLOCK_K: tl.constexpr,
                             USE_TOP_P: tl.constexpr, USE_FP64: tl.constexpr):
        # The image's _selector_walk_kernel (dflash2/speculator.py, SAMPLE_PROBABILISTIC=True) plus the draft scale
        # 1/tau and the draft-side nucleus; the noise is the same gumbel_noised_argmax call (same key, same keys).
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < top_k
        req_state = tl.load(req_state_ptr + row * num_steps)
        valid = req_state >= 0
        temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
        seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
        top_p = 1.0
        if USE_TOP_P:
            top_p = tl.load(top_p_ptr + req_state, mask=valid, other=1.0)
        previous = 0
        for step in range(num_steps):
            flat = row * num_steps + step
            score_base = (flat * top_k + previous) * top_k
            scores = tl.load(scores_ptr + score_base + offsets, mask=mask & valid,
                             other=float("-inf")).to(tl.float64 if USE_FP64 else tl.float32)
            candidate_base = flat * top_k
            candidates = tl.load(candidate_ptr + candidate_base + offsets, mask=mask & valid, other=0)
            tl.store(realized_scores_ptr + candidate_base + offsets, scores, mask=mask & valid)
            sel = scores
            if temperature != 0.0:
                sel = scores * inv_tau
                if USE_TOP_P:
                    z = (scores / temperature).to(tl.float32)
                    zmax = tl.max(z, axis=0)
                    e = tl.where(mask & valid, tl.exp(z - zmax), 0.0)
                    p = e / tl.sum(e, axis=0)
                    gt = z[:, None] > z[None, :]
                    above = tl.sum(tl.where(gt, p[:, None], 0.0), axis=0)
                    sel = tl.where(above < top_p, sel, float("-inf"))
            position = tl.load(sample_pos_ptr + flat) - 1
            _, index = gumbel_noised_argmax(sel, candidates, mask & valid, seed, position, temperature,
                                            USE_FP64=USE_FP64)
            token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
            tl.store(tokens_ptr + flat, token, mask=valid)
            previous = index

    _K["walk"] = _coupled_walk_kernel
    _K["triton"] = triton
    return _K


# ---------------------------------------------------------------------------------------------------
# state shared by the draft and the sampler side of one worker process
# ---------------------------------------------------------------------------------------------------
class Stats:
    steps = 0            # _verify calls with the coupled rule
    stock = 0            # _verify calls passed to the stock rule (mode 0)
    rows = 0             # verify rows (logits) under the coupled rule
    checked = 0          # steps compared with the stock Sampler (check mode)
    mismatch = 0         # steps with any committed / accepted / rejected / count mismatch (check mode)
    draw_mismatch = 0    # rows whose coupled draw != the stock Sampler draw (check mode; includes uncommitted rows)
    sampled_drafts = 0   # drafts verified on T > 0 rows (check mode)
    sampled_accepted = 0  # of which accepted (check mode)
    draft_graphs = {}    # capture variant (or "eager") -> coupled walk baked in?


_TOPP = {"buf": None, "np": None}


def _top_p_buffer(n: int, device=None):
    """Device top_p per request-state slot, read by the coupled walk inside the drafter graphs. Allocated once,
    eagerly, by Sampler.__init__ (before any capture: an allocation inside a capture would record its fill kernel into
    the graph); the captured graphs keep its address; written by Sampler.add_request through a fill kernel (async)."""
    import torch
    if _TOPP["buf"] is None:
        if device is None or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("glm-gumbel-coupled: top_p buffer not allocated before graph capture")
        _TOPP["buf"] = torch.ones(n, dtype=torch.float32, device=device)
    buf = _TOPP["buf"]
    if buf.numel() < n:
        raise RuntimeError(f"glm-gumbel-coupled: top_p buffer {buf.numel()} < {n} request slots")
    return buf


def _rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def stats() -> dict:
    s = Stats
    return {"steps": s.steps, "stock": s.stock, "rows": s.rows, "checked": s.checked, "mismatch": s.mismatch,
            "draw_mismatch": s.draw_mismatch, "sampled_drafts": s.sampled_drafts,
            "sampled_accepted": s.sampled_accepted}


def _maybe_log(force: bool = False) -> None:
    n = Stats.steps + Stats.stock
    if (force or (LOG_EVERY and n and n % LOG_EVERY == 0)) and _rank0():
        _log(f"steps coupled={Stats.steps} stock={Stats.stock} rows={Stats.rows} checked={Stats.checked} "
             f"mismatches={Stats.mismatch} draw_mismatch_rows={Stats.draw_mismatch} "
             f"T>0 drafts {Stats.sampled_drafts} accepted {Stats.sampled_accepted}")


# ---------------------------------------------------------------------------------------------------
# verify side
# ---------------------------------------------------------------------------------------------------
def coupled_verify(rs, logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np, expanded_idx_mapping,
                   expanded_local_pos):
    """RejectionSampler._verify with the coupled rule. Returns (processed_logits, sampled, num_sampled, target)."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    import glm_target_argmax as ta

    sampler = rs.sampler
    processed = sampler.apply_sampling_params(logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos,
                                              draft_sampled, expanded_local_pos)
    st = sampler.sampling_states
    target = gumbel_sample(processed, expanded_idx_mapping, st.temperature.gpu, st.seeds.gpu, pos,
                           apply_temperature=False, use_fp64=sampler.use_fp64_gumbel)
    num_reqs = cu_num_logits.shape[0] - 1
    sampled, num_sampled = ta.greedy_verify(target, draft_sampled, cu_num_logits, num_reqs,
                                            int(rs.num_speculative_steps) + 1)
    return processed, sampled, num_sampled, target


def check_against_stock(rs, logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                        expanded_idx_mapping, expanded_local_pos, sampled, num_sampled, target) -> dict:
    """Compare the coupled step with the STOCK Sampler.sample (its gumbel path) on the same logits rows."""
    import torch
    sampler = rs.sampler
    prev = sampler.use_flashinfer
    sampler.use_flashinfer = False
    try:
        ref, _ = sampler.sample(logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, draft_sampled,
                                expanded_local_pos, return_logprobs=False)
    finally:
        sampler.use_flashinfer = prev
    temp = sampler.sampling_states.temperature.gpu[expanded_idx_mapping.to(torch.int64)]
    return compare(ref.to(torch.int64), target.to(torch.int64), draft_sampled.to(torch.int64),
                   cu_num_logits.to(torch.int64), sampled.to(torch.int64), num_sampled.to(torch.int64), temp > 0)


def compare(ref, target, draft, cu, sampled, num_sampled, row_sampled) -> dict:
    """Pure tensor check (CPU-testable). ref [L]: stock draw per row; draft [L]: input ids at the logits rows (the
    draft checked at row i is draft[i + 1]); cu [R + 1]; sampled [R, W]; num_sampled [R]."""
    import torch
    dev = ref.device
    L = ref.numel()
    R = cu.numel() - 1
    rows = torch.arange(L, device=dev)
    req = torch.searchsorted(cu[1:].contiguous(), rows, right=True)
    local = rows - cu[req]
    nd = (cu[1:] - cu[:-1] - 1)[req]                                         # drafts of the row's request
    n = num_sampled[req]
    nxt = torch.cat([draft[1:], torch.full((1,), -1, dtype=draft.dtype, device=dev)])
    width = sampled.shape[1]
    committed = local < n
    got = sampled[req, local.clamp(max=width - 1)]
    bad_commit = committed & (got != ref)
    bad_accept = (local < n - 1) & (nxt != ref)
    bad_reject = (local == n - 1) & (local < nd) & (nxt == ref)
    ns = num_sampled
    ndr = cu[1:] - cu[:-1] - 1
    bad_count = (ns < 1) | (ns > ndr + 1)
    draw = target != ref
    is_draft_row = local < nd
    samp = row_sampled & is_draft_row & (nxt >= 0)
    return {"bad": int(bad_commit.sum() + bad_accept.sum() + bad_reject.sum() + bad_count.sum()),
            "draw": int(draw.sum()), "rows": L, "reqs": R,
            "sampled_drafts": int(samp.sum()), "sampled_accepted": int((samp & (local < n - 1)).sum())}


def install_rejection(mod) -> None:
    cls = mod.RejectionSampler
    if getattr(cls, "_glm_gc", False):
        return
    orig = cls._verify

    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                expanded_idx_mapping, expanded_local_pos):
        mode = _mode()
        if mode == "0":
            Stats.stock += 1
            _maybe_log()
            return orig(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                        expanded_idx_mapping, expanded_local_pos)
        if self.synthetic_conditional_rates is not None or self.use_block_verification:
            raise RuntimeError("glm-gumbel-coupled: synthetic / block verification cannot verify coupled drafts; "
                               "unset GLM_GUMBEL_COUPLED or use rejection_sample_method=standard")
        processed, sampled, num_sampled, target = coupled_verify(
            self, logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np, expanded_idx_mapping,
            expanded_local_pos)
        Stats.steps += 1
        Stats.rows += int(logits.shape[0])
        if mode == "check":
            res = check_against_stock(self, logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                                      expanded_idx_mapping, expanded_local_pos, sampled, num_sampled, target)
            Stats.checked += 1
            Stats.draw_mismatch += res["draw"]
            Stats.sampled_drafts += res["sampled_drafts"]
            Stats.sampled_accepted += res["sampled_accepted"]
            if res["bad"]:
                Stats.mismatch += 1
                if _rank0() and Stats.mismatch <= 20:
                    _log(f"MISMATCH #{Stats.mismatch}: {res} num_sampled={num_sampled.tolist()} "
                         f"cu={cu_num_logits.tolist()}")
        _maybe_log()
        return processed, sampled, num_sampled

    _verify.__wrapped__ = orig
    cls._verify = _verify
    cls._glm_gc = True
    _log(f"RejectionSampler._verify hooked (mode now {_mode()})")


# ---------------------------------------------------------------------------------------------------
# draft side
# ---------------------------------------------------------------------------------------------------
def _ab_guard() -> None:
    """Variants that differ on the coupled switch need per-variant drafter graphs (the walk is inside them)."""
    ab = sys.modules.get("glm_ab")
    if ab is None or not getattr(ab, "ACTIVE", False) or KEY not in ab.KNOWN:
        return
    vals = {ab.norm_value(KEY, ab.env_for(v, KEY)) != "0" for v in range(ab.N)}
    if len(vals) > 1 and not ab._opts.get("draft_sets"):
        raise RuntimeError("glm-gumbel-coupled: glm_ab variants differ on GLM_GUMBEL_COUPLED but the drafter graphs "
                           "are shared; set GLM_AB_DRAFT_SETS=1 (a stock verify of a coupled draft is not exact)")


def install_dflash2(mod) -> None:
    cls = mod.DFlash2Speculator
    if getattr(cls, "_glm_gc", False):
        return
    _ab_guard()
    orig = cls._sample_path
    warned = []

    def _sample_path(self, candidate_ids, scores, num_reqs):
        on = _mode() != "0"
        ab = sys.modules.get("glm_ab")
        tag = ab.current() if ab is not None and getattr(ab, "ACTIVE", False) else "static"
        Stats.draft_graphs[tag] = on
        if not on:
            return orig(self, candidate_ids, scores, num_reqs)
        if self.draft_logits is not None and not warned:
            warned.append(1)
            _log("draft_sample_method=probabilistic: the coupled walk replaces it (draft_logits are ignored by the "
                 "coupled verify)")
        k = _kernels()
        triton = k["triton"]
        top_p = _top_p_buffer(int(self.max_num_reqs))
        k["walk"][(num_reqs,)](
            scores.contiguous(), candidate_ids.contiguous(), self.sample_pos, self.sample_idx_mapping,
            self.temperature, self.seeds, top_p, self.draft_tokens, self._selector_scores, 1.0 / TAU,
            num_steps=self.num_speculative_steps, top_k=self.selector_top_k,
            BLOCK_K=triton.next_power_of_2(self.selector_top_k), USE_TOP_P=DRAFT_TOPP,
            USE_FP64=self.use_fp64_gumbel, num_warps=1)

    _sample_path.__wrapped__ = orig
    cls._sample_path = _sample_path
    cls._glm_gc = True
    _log(f"DFlash2Speculator._sample_path hooked (coupled walk: tau={TAU}, draft top-p={DRAFT_TOPP})")


def install_sampler(mod) -> None:
    """Sampler.__init__ allocates the walk's top_p buffer; Sampler.add_request mirrors each request's top_p into it
    (fill kernel, no sync)."""
    cls = mod.Sampler
    if getattr(cls, "_glm_gc", False):
        return
    orig = cls.add_request
    orig_init = cls.__init__

    def __init__(self, max_num_reqs, vocab_size, device, *args, **kwargs):
        orig_init(self, max_num_reqs, vocab_size, device, *args, **kwargs)
        _top_p_buffer(int(max_num_reqs), device)

    def add_request(self, req_idx, prompt_len, sampling_params):
        orig(self, req_idx, prompt_len, sampling_params)
        if DRAFT_TOPP:
            st = self.sampling_states
            _top_p_buffer(int(st.max_num_reqs))[req_idx].fill_(float(st.top_p.np[req_idx]))

    __init__.__wrapped__ = orig_init
    add_request.__wrapped__ = orig
    cls.__init__ = __init__
    cls.add_request = add_request
    cls._glm_gc = True


# ---------------------------------------------------------------------------------------------------
# drift guard + import hooks
# ---------------------------------------------------------------------------------------------------
def func_source(path: str, qualname: str) -> str:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    node = ast.parse(src)
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part:
                node = child
                break
        else:
            raise KeyError(f"{qualname} not found in {path}")
    lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    return textwrap.dedent("".join(lines))


def src_hash(root: str, key: str) -> str:
    mod, qual = key.split(":")
    path = os.path.join(root, *mod.split(".")) + ".py"
    return hashlib.sha256(func_source(path, qual).encode()).hexdigest()[:16]


def check_sources(root: str | None = None) -> None:
    if root is None:
        spec = importlib.util.find_spec("vllm")
        root = os.path.dirname(os.path.dirname(spec.origin))
    bad = []
    for key, want in EXPECTED.items():
        try:
            got = src_hash(root, key)
        except (OSError, KeyError) as exc:
            got = f"missing ({exc})"
        if got != want:
            bad.append(f"{key}: {got} != {want}")
    if bad:
        msg = "glm-gumbel-coupled: engine drift:\n  " + "\n  ".join(bad)
        if os.environ.get("GLM_GUMBEL_COUPLED_ALLOW_DRIFT") == "1":
            _log(msg + "\n(GLM_GUMBEL_COUPLED_ALLOW_DRIFT=1: patching anyway)")
        else:
            raise RuntimeError(msg + "\nrefusing to patch (GLM_GUMBEL_COUPLED_ALLOW_DRIFT=1 overrides)")


HOOKS = {RS: install_rejection, DF2: install_dflash2, SAMPLER: install_sampler}
_done: set = set()


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in HOOKS or name in _done:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=name):
            _orig(module)
            _install(_name, module)

        spec.loader.exec_module = exec_module
        return spec


def _install(name, module) -> None:
    if name in _done:
        return
    _done.add(name)
    HOOKS[name](module)


def register() -> None:
    """Called from overlay/sitecustomize.py when GLM_GUMBEL_COUPLED is set (or unioned on by glm_ab). Fatal on a
    drift or a bad configuration: a half-installed coupling (draft coupled, verify stock) would not be exact, so all
    three hooks install together or the process refuses to start."""
    try:
        check_sources()
        for name in list(HOOKS):
            if name in sys.modules:
                _install(name, sys.modules[name])
        if len(_done) < len(HOOKS) and not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except BaseException as exc:  # noqa: BLE001
        msg = f"glm-gumbel-coupled: refusing to start: {exc!r}"
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        os._exit(1)
