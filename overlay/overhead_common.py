"""Qualification helpers shared by the overlays and GPU harness."""


def bitwise_equal(a, b):
    import torch
    return (a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def disjoint_storage(tensors):
    # Conservative: reject even disjoint views of the same allocation. No GPU op.
    ptrs = [x.untyped_storage().data_ptr() for x in tensors]
    return len(set(ptrs)) == len(ptrs)
