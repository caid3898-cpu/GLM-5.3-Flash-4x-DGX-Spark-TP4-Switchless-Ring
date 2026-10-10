"""LVKP-S-L2 startup hooks. Diagnostic and unqualified registrations omitted.
Active registration statements preserve the accepted deployment's ordering.
"""
import importlib.abc
import importlib.util
import os
import sys

_orig = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_orig):
    try:
        spec = importlib.util.spec_from_file_location("_distro_sitecustomize", _orig)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pass

# In-boot A/B of adapter flags (overlay/glm_ab.py), TEST ONLY. Must run before every block below: it unions the
# switchable install gates into os.environ. Fatal on a config error. Inert unless GLM_AB_VARIANTS >= 2.
if os.environ.get("GLM_AB_VARIANTS", "").strip() not in ("", "0", "1"):
    import glm_ab
    for _k in ("GLM_L2_PREFETCH", "GLM_L2_PREFETCH_AR", "GLM_KDA_STASH_NOCOPY", "GLM_KDA_FLAG_FUSED",
               "GLM_DFLASH_CTX_GRAPH", "GLM_ROUTER_DEDUP", "GLM_MHC_FUSED", "GLM_EARLY_PLAN",
               "GLM_L2_PREFETCH_MOE"):
        glm_ab.KNOWN.setdefault(_k, "bool")
    for _k in ("GLM_DENSE_FAST", "GLM_DENSE_FAST_PREFILL"):   # 0 | 1 | comma list of shape names
        glm_ab.KNOWN.setdefault(_k, "raw")
    # 2026-09-28 speed screen (diagnostics/glm-speedscreen-20260928): windows M / B-MLA / D and the BF16 hc fn read
    # (glm-bytes-20260928), the Marlin launch table (glm-marlin-tune-20260928; install gate GLM_MARLIN_TUNE=<path>,
    # per-variant switch GLM_MARLIN_TUNE_ON), the fused drafter conv (glm-megagraph-20260928; drafter graphs, needs
    # GLM_AB_DRAFT_SETS=1) and the RoCE proxy thread pin (glm-comm-20260928; host-side, applied by a switch hook).
    for _k in ("GLM_L2_PREFETCH_MLA", "GLM_L2_PREFETCH_MLA_AR", "GLM_L2_PREFETCH_DRAFT", "GLM_MHC_BF16W",
               "GLM_MARLIN_TUNE_ON"):
        glm_ab.KNOWN.setdefault(_k, "bool")
    for _k in ("GLM_L2_PREFETCH_MLA_MB", "GLM_L2_PREFETCH_DRAFT_MB", "GLM_ROCE_PROXY_CPUS"):
        glm_ab.KNOWN.setdefault(_k, "raw")
    glm_ab.KNOWN.setdefault("GLM_DRAFT_CONV_FUSED", "mode")
    glm_ab.HASHED_SOURCES = tuple(glm_ab.HASHED_SOURCES) + ("glm_l2_prefetch", "glm_kda_stash_fast",
                                                            "glm_dflash_ctx_graph", "glm_router_dedup",
                                                            "glm_mhc_hook", "mhc_fused", "glm_early_plan",
                                                            "glm_l2_prefetch_c", "glm_dense_fast",
                                                            "glm_l2_prefetch_mla", "glm_mhc_bf16w",
                                                            "glm_marlin_tune", "glm_draft_conv_fused",
                                                            "glm_roce_proxy_pin")
    # 2026-09-28 final stack (diagnostics/glm-final-20260928): the prefill package pf2 (glm-prefill2-20260928; mHC
    # prefill sharding + padding, KDA conv split, FlashKDA, Triton sparse MLA: all eager, read per forward),
    # Gumbel-coupled drafting (glm-gumbel-20260928; draft walk inside the drafter graphs) and the draft-shape
    # truncation (glm-draft-trunc-20260928). Only used by the in-boot A/B harness.
    glm_ab.KNOWN.setdefault("GLM_PREFILL_SHARD", "raw")
    for _k in ("GLM_PREFILL_SHARD_PAD", "GLM_KDA_CONV_SPLIT", "GLM_FLASHKDA_PREFILL", "GLM_TRITON_MLA_PREFILL"):
        glm_ab.KNOWN.setdefault(_k, "bool")
    glm_ab.KNOWN.setdefault("GLM_GUMBEL_COUPLED", "mode")
    glm_ab.KNOWN.setdefault("GLM_DRAFT_TRUNC", "raw")
    # 2026-09-29 prefill3 (diagnostics/glm-prefill3-20260928): routed-MoE prefill sum+add / down tile / fused act
    for _k in ("GLM_PF3_SUMADD", "GLM_PF3_DOWN", "GLM_PF3_ACT"):
        glm_ab.KNOWN.setdefault(_k, "bool")
    glm_ab.HASHED_SOURCES = tuple(glm_ab.HASHED_SOURCES) + ("glm_pf3_moe", "glm_pf3_jit")
    # 2026-09-29 final3 (diagnostics/glm-final-20260928/FINAL3.md): large dim-0 all-gathers to NCCL (glm-prefill-overlap)
    glm_ab.KNOWN.setdefault("GLM_ROCE_AG_DIM0_NCCL", "bool")
    glm_ab.HASHED_SOURCES = tuple(glm_ab.HASHED_SOURCES) + ("glm_roce_gather_route",)
    glm_ab.HASHED_SOURCES = tuple(glm_ab.HASHED_SOURCES) + ("glm_prefill_shard", "glm_prefill_hooks",
                                                            "glm_kda_conv_split", "glm_flashkda",
                                                            "glm_sparse_mla_prefill", "glm_sparse_mla_kernel",
                                                            "glm_gumbel_coupled", "glm_draft_trunc")
    glm_ab.KNOWN.setdefault("GLM_DEVSELECT", "raw")   # device-side verify-shape selection
    glm_ab.HASHED_SOURCES = tuple(glm_ab.HASHED_SOURCES) + ("glm_devselect",)
    glm_ab.register()

_POST = {}  # module name -> list of callables(module)




if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_LV_CGSTAT", "GLM_LV_SPLIT_FIX", "GLM_LV_ARGMAX_MINTOK", "GLM_LV_DRAFT_FP8_KV")):
    try:
        import glm_levers as _glv
        _glv.register_early()
        for _name, _fn in _glv.HOOKS.items():
            _POST.setdefault(_name, []).append(_fn)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-levers: import failed: %r\n" % (exc,))

if _POST:

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _POST:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module
            fns = _POST.pop(name)

            def patched_exec(module):
                exec_module(module)
                for fn in fns:
                    try:
                        fn(module)
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write("overlay hook for %s failed: %r\n" % (name, exc))
                        raise

            loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _Hook())


if any(os.environ.get(k) == "1" for k in ("QMIX_FP8_BLOCK", "QMIX_DRAFT_HEAD_FP8", "QMIX_DEBUG_LAYERS")):
    import qmix_patch
    qmix_patch.register()

if os.environ.get("GLM_FAST_LOAD", "0").strip().lower() in ("1", "on", "true"):
    import glm_fast_load
    glm_fast_load.register()
if any(k.startswith("GLM_DS_") and os.environ[k].strip() not in ("", "0", "off") for k in os.environ):
    import glm_ds_hooks
    glm_ds_hooks.register()

# --- Certified target head + 4-bit drafter (diagnostics/glm-head-drafter-20260928) ---
# Default OFF. After the glm_ds_hooks block (both reuse its after-import finder); order-independent with
# glm_exact_hooks / glm_levers. GLM_CERT_HEAD needs GLM_TARGET_VOCAB_ARGMAX=1. A drift in the engine sources raises.
if os.environ.get("GLM_CERT_HEAD", "0").strip().lower() not in ("", "0", "off", "false", "no") \
        or os.environ.get("GLM_CERT_HEAD_DUMP", "").strip():
    import glm_cert_head
    glm_cert_head.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_DS_DRAFT_NVFP4", "GLM_DS_DRAFT_HEAD_FP4", "GLM_CERT_HEAD_DRAFT")):
    import glm_ds_draft_fp4
    glm_ds_draft_fp4.register()

# --- Context-lookup drafter (overlay/glm_lookup_hooks.py + glm_lookup_draft.py; diagnostics/glm-lookup-20260928) ---
# Default OFF. GLM_LOOKUP_DRAFT=1 patches the V2 runner and the DFlash propose on the workers; the plans come
# from --scheduler-cls glm_lookup_draft.LookupScheduler (profile SCHEDULER_CLS). GLM_LOOKUP_DRAFT=shadow needs
# only the scheduler class (index + counterfactual accounting, drafts unchanged). Place after the glm_ds_hooks
# block (this module reuses its after-import finder and drift-guard helpers). A failing install raises.
if os.environ.get("GLM_LOOKUP_DRAFT", "0").strip().lower() not in ("", "0", "off", "false", "no", "shadow"):
    import glm_lookup_hooks
    glm_lookup_hooks.register()

if "1" in (os.environ.get("GLM_KDA_STASH"), os.environ.get("GLM_ROUTER_FP32OUT")):
    import glm_kda_stash
    glm_kda_stash.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_TARGET_VOCAB_ARGMAX", "GLM_KDA_NOCOPY")):
    import glm_exact_hooks
    glm_exact_hooks.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_KDA_STASH_NOCOPY", "GLM_KDA_FLAG_FUSED")):
    import glm_kda_stash_fast
    glm_kda_stash_fast.register()
if os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_l2_prefetch
    glm_l2_prefetch.register()
# Default-off candidate (diagnostics/glm-steplevel-20260928): window C, the MoE/dense-MLP output all-reduce
# prefetches the next layer's hc_attn_fn + first attention projection. Needs the two lines above.
if os.environ.get("GLM_L2_PREFETCH_MOE", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_l2_prefetch_c
        glm_l2_prefetch_c.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-l2-prefetch-c: import failed: %r\n" % (exc,))

# Default-off candidate registration; source identical to the scored diagnostic.
# Router dedup: JaredforReal, vLLM #55736. Integer GDN producer: vLLM metadata builder.
if os.environ.get("GLM_ROUTER_DEDUP", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_router_dedup
    glm_router_dedup.register()
# Default-off candidate: split-K Triton GEMMs for the DSA indexer head gate and the MoE router logits
# (diagnostics/glm-gemv-20260928). Registered after glm_kda_stash and glm_router_dedup.
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_GATE_GEMV", "GLM_ROUTER_GEMV")):
    try:
        import glm_small_gemv
        glm_small_gemv.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-small-gemv: import failed: %r\n" % (exc,))
# Default-off candidate (diagnostics/glm-cuda-kernels-20260928 + glm-prefill-kernels-20260928): the dense 8-bit
# Marlin linears (KDA in_proj / o MXFP8, MLA + shared-expert block FP8) dispatched per (shape, M) from
# overlay/glm_dense_fast_table.json: GLM_DENSE_FAST = decode kernel (M <= 32), GLM_DENSE_FAST_PREFILL = unpack +
# cuBLAS (large M). Numerics-changing (fp32 order): KLD-gate it. JIT build + self-test + TP agreement at load;
# fails closed to stock. A config error (unknown shape name, bad table) is fatal here.
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_DENSE_FAST", "GLM_DENSE_FAST_PREFILL")):
    import glm_dense_fast
    glm_dense_fast.register()
if os.environ.get("GLM_GDN_METADATA_FAST", "0") == "1":
    import glm_gdn_hook
    glm_gdn_hook.register()

# Multimodal long-side ceiling (glm-mm-resize-20261008). GLM_MM_MAX_SIDE px, 0/off = inert.
# --limit-mm-per-prompt bounds the image COUNT and max_image_tokens bounds one image's TOKEN
# cost, but neither bounds the pixels the vision encoder decodes and holds: every image's
# pixel_values stay resident as a full bfloat16 tensor, so 32 x 12 MP killed the worker
# (exit code: None, a signal, not a Python exception). This clamps the long side before the
# processor's smart_resize picks a canvas, shrinking both the token cost and the encoder
# working set. Aspect ratio preserved; images within the limit pass through untouched.
if os.environ.get("GLM_MM_MAX_SIDE", "").strip():
    try:
        import glm_mm_resize
        glm_mm_resize.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-mm-resize: import failed: %r\n" % (exc,))

# Default-off: shorter shm queue busy-wait for readers (nacyot, GB10); output unchanged.
if os.environ.get("GLM_SHM_BUSY_LOOP_S", "").strip():
    import glm_shm_spin
    glm_shm_spin.register()

# Default-off candidate (diagnostics/glm-comm-20260928): pin RoCEnante's RDMA proxy thread to one performance core
# (GLM_ROCE_PROXY_CPUS=auto|<list>). Thread placement only; output unchanged.
if os.environ.get("GLM_ROCE_PROXY_CPUS", "").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_roce_proxy_pin
        glm_roce_proxy_pin.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-roce-proxy-pin: register failed: %r\n" % (exc,))

# Default-off candidate: exact sync-free kv_lens. Cheap flag pre-check so
# the disabled default does not import torch at interpreter startup.
if (
    os.environ.get("GLM_KVLENS_EXACT", "").strip().lower()
        not in ("", "0", "off", "false", "no")
    or os.path.exists("/overlay/overlay/glm_kvlens_exact.enabled")
):
    try:
        import glm_kv_lens_exact as _gkv
        _gkv.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-kvlens-exact: import failed: %r\n" % (exc,))

# Default-off candidate (diagnostics/glm-steplevel-20260928): the SM90 planner syncs on the verify-time
# num_computed_tokens snapshot instead of the post-draft positions copy, so the host prepares and launches the
# next target graph while the drafter runs. Exact; refuses to install together with GLM_KVLENS_EXACT.
if os.environ.get("GLM_EARLY_PLAN", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_early_plan
        glm_early_plan.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-early-plan: import failed: %r\n" % (exc,))

# Default-off candidate (diagnostics/glm-steplevel-20260928): bias-corrected warm start of the adaptive-k EMA
# (GLM_LV_WARM=<prior weight>, 1 intended). Engine-core side only.
if os.environ.get("GLM_LV_WARM", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_k_warm
        glm_k_warm.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-k-warm: import failed: %r\n" % (exc,))

# Default-off candidate: forced fused_marlin_moe M-tile (decode MoE sweep,
# glm-window-20260927 night loop). Env-only arm; no marker files.
if os.environ.get("GLM_MARLIN_MOE_BLOCK_M", "0").strip() not in ("", "0", "off", "false", "no"):
    try:
        import glm_marlin_m32
        glm_marlin_m32.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-marlin-m32: import failed: %r\n" % (exc,))

# Default-off candidate (diagnostics/glm-marlin-tune-20260928): per-shape Marlin launch configs + MoE persistent
# workspace (GLM_MARLIN_TUNE=<tune.json>, GLM_MARLIN_TUNE_SHA=<sha256>). Env-only arm; no torch import when unset.
if os.environ.get("GLM_MARLIN_TUNE", "").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_marlin_tune
        glm_marlin_tune.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-marlin-tune: register failed: %r\n" % (exc,))

# Default-off candidate: DFlash context-KV precompute replayed from
# exact-shape CUDA graphs (glm-window-20260927 night loop, t<->d glue).
if os.environ.get("GLM_DFLASH_CTX_GRAPH", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_dflash_ctx_graph
        glm_dflash_ctx_graph.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-dflash-ctx-graph: import failed: %r\n" % (exc,))
# Default-off candidate (diagnostics/glm-megagraph-20260928): the DFlash2 drafter's grouped short convolution
# (qwen3_dflash2._grouped_conv, 10 eager kernels x 20 calls per draft step) as one bit-exact Triton kernel per call.
if os.environ.get("GLM_DRAFT_CONV_FUSED", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    try:
        import glm_draft_conv_fused
        glm_draft_conv_fused.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-draft-conv-fused: import failed: %r\n" % (exc,))
#   GLM_MHC_FUSED=1          bf16-load mHC projection kernels, bit-exact
#                            (overlay/mhc_fused.py + glm_mhc_hook.py)
if os.environ.get("GLM_MHC_FUSED", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_mhc_hook
    glm_mhc_hook.register()

# --- GLM prefill adapters (overlay/glm_prefill_hooks.py; diagnostics/glm-prefill) --------------------
# Inert unless GLM_PREFILL_SHARD / GLM_PREFILL_CADENCE / GLM_PREFILL_SHORT_TOKENS /
# GLM_PREFILL_CADENCE_WHEN_QUEUED / GLM_END_DRAIN / GLM_IDLE_COALESCE_MS is set to a non-zero value.
if any(k.startswith(("GLM_PREFILL_", "GLM_END_DRAIN", "GLM_IDLE_COALESCE")) for k in os.environ):
    import glm_prefill_hooks
    glm_prefill_hooks.register()

# --- KDA state checkpoints at 2304-token block ends (overlay/glm_mamba_align_fix.py; issue #2) ---------------
# Correctness fix, on unless GLM_MAMBA_ALIGN_FIX=0: prefill chunks end on mamba_block_size, and the last full
# block is materialized, so a prefix-cache hit restores the state its hash claims.
if os.environ.get("GLM_MAMBA_ALIGN_FIX", "1").strip() != "0":
    import glm_mamba_align_fix
    glm_mamba_align_fix.register()

# ---- glm-bytes-20260928 (default-off candidates): windows M / B-MLA / D (GLM_L2_PREFETCH_MLA / _MLA_AR / _DRAFT,
# on top of GLM_L2_PREFETCH{,_AR}=1) and the BF16 hc fn read (GLM_MHC_BF16W). At the END: after glm_l2_prefetch / _c
# and after glm_small_gemv, whose Indexer source rewrite must run before window M wraps Indexer.forward.
_on = lambda k: os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")  # noqa: E731
if _on("GLM_L2_PREFETCH") and any(_on(k) for k in ("GLM_L2_PREFETCH_MLA", "GLM_L2_PREFETCH_MLA_AR",
                                                    "GLM_L2_PREFETCH_DRAFT")):
    try:
        import glm_l2_prefetch_mla
        glm_l2_prefetch_mla.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-l2-prefetch-mla: import failed: %r\n" % (exc,))
if _on("GLM_MHC_BF16W"):
    if _on("GLM_MHC_FUSED"):
        sys.stderr.write("glm-mhc-bf16w: GLM_MHC_FUSED is also set; both patch mhc_fused_tilelang, bf16w not loaded\n")
    else:
        try:
            import glm_mhc_bf16w
            glm_mhc_bf16w.register()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("glm-mhc-bf16w: import failed: %r\n" % (exc,))

# ==== 2026-09-28 final stack (diagnostics/glm-final-20260928): blocks merged from the prefill2, gumbel and trunc
# overlays, in the order their own trees registered them (after every block above).

# --- KDA prefill short conv as three dense outputs (overlay/glm_kda_conv_split.py; diagnostics/glm-prefill2-20260928).
# Default OFF. Wraps causal_conv1d_fn / chunk_kda_with_fused_gate in the kda module (keeps itself outermost on the
# chunk name); _forward itself is untouched (hash-pinned by glm_exact_hooks). Idea: mmastrac (conv out_group).
if os.environ.get("GLM_KDA_CONV_SPLIT", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_kda_conv_split
    glm_kda_conv_split.register()

# --- FlashKDA for the KDA chunked prefill (overlay/glm_flashkda.py; diagnostics/glm-flashkda-20260928) -----------
# Default OFF. Wraps chunk_kda_with_fused_gate (chains to the previous value) + a raw-beta tap on _cast_sigmoid in
# the kda module; _forward untouched. JIT build + load-time self-test, TP-agreed; fails closed to stock.
# Numerics-changing: KLD-gate it. FlashKDA: MoonshotAI (MIT), mmastrac wip-fp32-state; call per vLLM #55737.
if os.environ.get("GLM_FLASHKDA_PREFILL", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_flashkda
    glm_flashkda.register()

# --- Triton sparse MLA, eager prefill/mixed steps (overlay/glm_sparse_mla_prefill.py; diagnostics/glm-tmla-20260928) ---
# Default OFF. GLM_TRITON_MLA_PREFILL=1 wraps FlashInferMLASparseSM90Impl.forward_mqa: steps with prefill rows run one
# Triton kernel (top-k fp8 latent rows gathered in-kernel, online softmax); decode graphs keep FlashInfer.
# Numerics-changing (summation order): KLD-gate it. Idea: mmastrac (PR #4), repro chuck-ads.
if os.environ.get("GLM_TRITON_MLA_PREFILL", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_sparse_mla_prefill
    glm_sparse_mla_prefill.register()

# Gumbel-coupled drafting at T > 0 (overlay/glm_gumbel_coupled.py; port of llama.cpp-lab PR #26 by Jim Routh), exact.
# Default OFF: GLM_GUMBEL_COUPLED=1|check. Refuses to start on engine drift or a half-installed coupling.
if os.environ.get("GLM_GUMBEL_COUPLED", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_gumbel_coupled
    glm_gumbel_coupled.register()

# Draft-shape truncation: choose the real verify shape from the current draft (GLM_DRAFT_TRUNC; registered last, as in
# the 2209 A/B tree: its DFlash2 propose / runner hooks wrap whatever the blocks above installed).
if os.environ.get("GLM_DRAFT_TRUNC", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_draft_trunc
    glm_draft_trunc.register()

# --- prefill3 routed-MoE kernels (overlay/glm_pf3_moe.py; diagnostics/glm-prefill3-20260928) --------------------
# Default OFF. GLM_PF3_SUMADD (exact fused moe_sum + shared add), GLM_PF3_DOWN / GLM_PF3_ACT (same-math Marlin down
# tile / gate_up+SiLU epilogue), eager prefill only, per call through glm_ab. Hooks marlin_moe._fused_marlin_moe,
# MarlinExperts.moe_sum, moe_runner._unpack and MoERunner._apply_quant_method (after-import, chains).
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_PF3_SUMADD", "GLM_PF3_DOWN", "GLM_PF3_ACT")):
    import glm_pf3_moe
    glm_pf3_moe.register()

# --- final3: large dim-0 all-gathers via NCCL (overlay/glm_roce_gather_route.py; diagnostics/glm-prefill-overlap-20260928)
# Default OFF. GLM_ROCE_AG_DIM0_NCCL=1 wraps glm_roce.adapter.GlmRoceAllReduce.should_all_gather: dim-0 gathers whose
# per-rank shard exceeds GLM_ROCE_AG_DIM0_NCCL_ABOVE (4MiB) take the stock NCCL gather (the mHC prefill-shard row
# gathers, -20 % each); last-dim gathers (logits) and GLM_ROCE_GATHER_MAX_SIZE are unchanged. Exact (a copy).
if os.environ.get("GLM_ROCE_AG_DIM0_NCCL", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_roce_gather_route
    glm_roce_gather_route.register()

# --- corrected window B / C L2 prefetch tables (overlay/glm_l2pf_v2.py; diagnostics/glm-tinykernels-20260928) -------
# Default OFF. GLM_L2PF_HC=1: windows B and C warm the hc tensor the mHC kernel actually reads (its BF16 twin while
# glm_mhc_bf16w is live) instead of the unused FP32 parameter. Exact (cache hints). Registered last: it wraps
# glm_l2_prefetch / glm_l2_prefetch_c functions registered above. An import error leaves the stock tables.
if any(_on(k) for k in ("GLM_L2PF_HC", "GLM_L2PF_ROUTER")):
    try:
        import glm_l2pf_v2
        glm_l2pf_v2.register()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-l2pf-v2: import failed: %r\n" % (exc,))
