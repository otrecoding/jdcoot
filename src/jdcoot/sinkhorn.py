"""Iterative scaling of a positive kernel to prescribed marginals."""

import numpy as np


def sinkhorn_scaling(
    a,
    b,
    K,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    always_raise=False,
    **kwargs,
):
    """Scale K to marginals a and b, optionally returning convergence details.

    With multiple target histograms (columns of b), return each scaled
    kernel's total mass instead of a coupling matrix. The legacy
    ``always_raise`` and extra keyword arguments are currently unused.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    n_source = len(a)
    n_target = len(b)

    nbb = b.shape[1] if b.ndim > 1 else 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of distances
    if nbb:
        u = np.ones((n_source, nbb)) / n_source
        v = np.ones((n_target, nbb)) / n_target
    else:
        u = np.ones(n_source) / n_source
        v = np.ones(n_target) / n_target

    scaled_kernel = (1 / a[:, np.newaxis]) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        uprev = u
        vprev = v
        kernel_transpose_u = np.dot(K.T, u)
        v = np.divide(b, kernel_transpose_u)
        u = 1.0 / np.dot(scaled_kernel, v)

        zero_in_transp = np.any(kernel_transpose_u == 0)
        nan_in_dual = np.any(np.isnan(u)) or np.any(np.isnan(v))
        inf_in_dual = np.any(np.isinf(u)) or np.any(np.isinf(v))
        if zero_in_transp or nan_in_dual or inf_in_dual:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print("Warning: numerical errors at iteration in sinkhorn_scaling", cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum(
                    (v - vprev) ** 2
                ) / np.sum((v) ** 2)
            else:
                transp = u[:, np.newaxis] * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("It.  | Err" + "\n" + "-" * 19)
                print(f"{cpt:5d}|{err:8e}|")
        cpt += 1
    if log:
        log["u"] = u
        log["v"] = v
    if nbb:  # return only loss
        res = np.sum(u.T[:, :, None] * K * v.T[:, None, :], axis=(1, 2))
        if log:
            return res, log
        return res

    if log:
        return u[:, np.newaxis] * K * v[np.newaxis, :], log
    return u[:, np.newaxis] * K * v[np.newaxis, :]
