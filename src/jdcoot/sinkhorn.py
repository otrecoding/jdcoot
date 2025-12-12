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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    Nini = len(a)
    Nfin = len(b)

    nbb = b.shape[1] if b.ndim > 1 else 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    Kp = (1 / a[:, np.newaxis]) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1.0 / np.dot(Kp, v)

        zero_in_transp = np.any(KtransposeU == 0)
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
    M = 1  # FIXME: add by Pierre to fix the undefined variable M
    if nbb:  # return only loss
        res = np.sum(u.T[:, :, None] * K * v.T[:, None, :] * M, axis=(1, 2))
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u[:, np.newaxis] * K * v[np.newaxis, :], log
        else:
            return u[:, np.newaxis] * K * v[np.newaxis, :]
