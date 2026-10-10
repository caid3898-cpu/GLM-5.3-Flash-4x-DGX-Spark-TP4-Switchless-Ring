# SPDX-License-Identifier: Apache-2.0
"""Default-off integration of vLLM's fused integer GDN metadata builder.

GLM_GDN_METADATA_FAST=1 enables the locally byte-qualified producer. A bounded
startup check compares all metadata fields; failed checks abort, never fall back.
Credit: vLLM GDNAttentionMetadataBuilder and overhead/glm_gdn_metadata_fast.py.
"""
import functools
import os
import sys

TARGET = 'vllm.v1.attention.backends.gdn_attn'
COUNTS = dict(fast=0, fallback=0, checked=0)


def enabled():
    ab = sys.modules.get('glm_ab')
    get = ab.env if ab is not None and getattr(ab, 'ACTIVE', False) else os.environ.get
    return get('GLM_GDN_METADATA_FAST', '0') == '1'


def install(module):
    cls = module.GDNAttentionMetadataBuilder
    if getattr(cls, '_glm_gdn_fast_installed', False):
        return
    stock = cls.build

    @functools.wraps(stock)
    def build(self, common_prefix_len, common_attn_metadata,
              num_accepted_tokens=None, num_decode_draft_tokens_cpu=None,
              fast_build=False):
        args = (self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
        if not enabled():
            return stock(*args)
        from glm_gdn_metadata_fast import try_build
        result = try_build(self, common_attn_metadata,
                           num_accepted_tokens, num_decode_draft_tokens_cpu)
        if result is None:
            COUNTS['fallback'] += 1
            return stock(*args)
        COUNTS['fast'] += 1
        # Per-builder checks cover each layer group; set to zero only after a
        # separately recorded checking warmup, before measurement/graph capture.
        limit = int(os.environ.get('GLM_GDN_METADATA_CHECK_CALLS', '8'))
        checked = getattr(self, '_glm_gdn_checked', 0)
        if checked < limit:
            import torch
            from overhead_common import bitwise_equal
            snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in vars(result).items()}
            ref = stock(*args)
            if snapshot.keys() != vars(ref).keys():
                raise RuntimeError('GDN metadata field set mismatch')
            for name, actual in vars(ref).items():
                expected = snapshot[name]
                same = (bitwise_equal(expected, actual) if isinstance(expected, torch.Tensor)
                        else expected == actual)
                if not same:
                    raise RuntimeError(f'GDN metadata mismatch: {name}')
            # Stock writes the same persistent outputs during this check.
            self._glm_gdn_checked = checked + 1
            COUNTS['checked'] += 1
            if checked == 0:
                print('glm-gdn-fast: metadata bytes checked on live inputs', file=sys.stderr, flush=True)
        return result

    cls.build = build
    cls._glm_gdn_fast_installed = True


def register():
    import importlib.abc
    import importlib.util
    if TARGET in sys.modules:
        install(sys.modules[TARGET])
        return

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                install(module)
            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Finder())
