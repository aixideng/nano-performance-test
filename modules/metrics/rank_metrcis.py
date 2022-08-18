import numpy as np


def mean_reciprocal_rank(rs):
    """Reciprocal of the rank of the first relevant item (MRR)"""
    
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_at_k(r, k):
    """Precision @ k"""
    
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    """Average precision (AP)"""
    
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Mean average precision (MAP)"""
    
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Discounted cumulative gain (DCG)"""
    
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("Param 'method' must be 0 or 1.")
    return 0.


def ndcg_at_k(r, k, method=0):
    """Normalized discounted cumulative gain (NDCG)"""
    
    dcg_max = dcg_at_k(np.sort(r)[::-1], k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
