import numpy as np
import ot
from ..comp import comp_regression
from ..utils import xcolumns, continuous_classifiers, continuous_accuracy
from ..coot import init_matrix_np


def _normalize(M, eps=1e-12):
    """Scale a cost matrix so it's comparable in magnitude to the others
    being summed into Ms. Prevents one term (e.g. the label cost) from
    silently dominating just because of arbitrary units."""
    m = M.max()
    return M / (m + eps) if m > 0 else M


def _mask_unknown_labels(M_lin, unknown_idx, axis):
    """Replace the columns/rows of M_lin that correspond to unlabeled
    (sentinel -1) points with the mean of the *known* entries.

    Without this, unknown labels get compared numerically against real
    labels (e.g. distance-to -1), injecting a fake but nonzero cost that
    biases the OT coupling for exactly the points we have no label
    information about. Neutralizing them lets the feature/structure term
    (C_s) drive the coupling for those points instead, which is what
    should happen when there's no label signal.
    """
    if len(unknown_idx) == 0:
        return M_lin
    n = M_lin.shape[axis]
    known_idx = np.setdiff1d(np.arange(n), unknown_idx)
    if axis == 1:
        fill = M_lin[:, known_idx].mean() if len(known_idx) else 0.0
        M_lin[:, unknown_idx] = fill
    else:
        fill = M_lin[known_idx, :].mean() if len(known_idx) else 0.0
        M_lin[unknown_idx, :] = fill
    return M_lin


def _run_bcd_block(
    x_anchor_train,
    y_anchor_train,
    x_full,
    y_full_train_known,
    l_full_train,
    l_full_test,
    clf_full,
    M_lin,
    algo1,
    algo2,
    reg,
    reg2,
    alpha_label_uplift,
    alphaLin,
    beta_fcost,
    normalize,
    batch_size,
    nb_epoch,
    numIterBCD,
    perf_reference,
    verbose,
):
    """One directional BCD block (source->target or target->source),
    shared so both halves of the algorithm get identical fixes and
    can't drift out of sync.

    perf_reference: the ground-truth (or masked) Y series for the "full"
    domain, used only to print/track diagnostics -- never used to leak
    test labels into training.
    """
    nA, dA = x_anchor_train.shape
    nB, dB = x_full.shape

    vA = np.ones(dA) / dA
    vB = np.ones(dB) / dB
    wA = np.ones(nA) / nA
    wB = np.ones(nB) / nB

    C_s, h1_s, h2_s = init_matrix_np(x_anchor_train, x_full, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_anchor_train.T, x_full.T, wA, wB)

    Gs = np.ones((nA, nB)) / (nA * nB)
    Gv = np.ones((dA, dB)) / (dA * dB)

    M_lin_n = _normalize(M_lin) if normalize else M_lin

    fcost = M_lin_n.copy()
    cost = np.inf

    best_cost = np.inf
    best_weights = clf_full.get_weights()
    best_iter = -1

    for k in range(numIterBCD):
        costold = cost
        Gsold = Gs.copy()
        Gvold = Gv.copy()

        C_s_n = _normalize(C_s) if normalize else C_s
        fcost_n = _normalize(fcost) if normalize else fcost

        # step 1: samples coupling optimization
        Ms = (
            (C_s_n - np.dot(h1_s, Gv).dot(h2_s.T))
            + alphaLin * M_lin_n
            + alpha_label_uplift * fcost_n
        )
        if algo1 == "emd":
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo1 == "sinkhorn":
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2: features coupling optimization
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)
        if algo2 == "emd":
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        y_full_hat = nB * Gs.T.dot(y_anchor_train.reshape(-1, 1))
        y_full_hat[l_full_train] = y_full_train_known

        clf_full.fit(
            x_full, y_full_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        y_full_pred = clf_full.predict(x_full, verbose=0).ravel()
        y_full_pred[l_full_train] = y_full_train_known.ravel()

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        perf = continuous_accuracy(y_full_pred.ravel(), perf_reference)

        if verbose:
            print(f"Delta: {delta} \t Loss: {cost} \t Accuracy: {perf}")

        # checkpoint the best model seen so far, judged by the OT cost
        # alone (unsupervised, so this never touches held-out labels)
        if cost < best_cost:
            best_cost = cost
            best_weights = clf_full.get_weights()
            best_iter = k

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print("converged at iter ", k)
            break

        new_fcost = ot.dist(
            y_anchor_train.reshape(-1, 1),
            y_full_pred.reshape(-1, 1),
            metric="sqeuclidean",
        )
        new_fcost_n = _normalize(new_fcost) if normalize else new_fcost
        # exponential smoothing damps the OT<->classifier feedback loop
        # instead of letting each iteration fully overwrite the last
        fcost = beta_fcost * fcost_n + (1 - beta_fcost) * new_fcost_n

    if verbose:
        print(f"restoring best checkpoint from iter {best_iter} (cost={best_cost})")
    clf_full.set_weights(best_weights)
    return clf_full


def continuous_partial_jdcoot(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
    **kwargs,
):
    alphaS = kwargs.get("alphaS", 2.425)
    alphaT = kwargs.get("alphaT", 2.425)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    nb_epoch = kwargs.get("nb_epoch", 10)  # was hardcoded before, kwarg was dead
    algo1 = algo
    algo2 = "emd"
    reg2 = 0
    numIterBCD = kwargs.get("numIterBCD", 100)

    # new, tunable knobs -- all default to sensible values so behavior
    # is a strict improvement without requiring any call-site changes
    alphaLin = kwargs.get("alphaLin", 1.0)
    beta_fcost = kwargs.get("beta_fcost", 0.3)
    normalize = kwargs.get("normalize_costs", True)
    verbose = kwargs.get("verbose", True)

    def compute_cost_matrix(ys, yt):
        return ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    y_source = source.Y.values[:, np.newaxis]
    y_target = target.Y.values[:, np.newaxis]

    clf_source, clf_target = continuous_classifiers(source, target)

    x_source_train = x_source[l_source_train, :]
    y_source_train = y_source[l_source_train, :]
    x_target_train = x_target[l_target_train, :]
    y_target_train = y_target[l_target_train, :]

    # ---- block 1: source(train) -> target(all) ----
    y_target2 = y_target.copy()
    y_target2[l_target_test] = -1
    y_source2 = y_source_train.copy()
    M_lin_1 = compute_cost_matrix(yt=y_target2, ys=y_source2)
    M_lin_1 = _mask_unknown_labels(M_lin_1, np.asarray(l_target_test), axis=1)

    clf_target = _run_bcd_block(
        x_anchor_train=x_source_train,
        y_anchor_train=y_source_train,
        x_full=x_target,
        y_full_train_known=y_target_train,
        l_full_train=l_target_train,
        l_full_test=l_target_test,
        clf_full=clf_target,
        M_lin=M_lin_1,
        algo1=algo1,
        algo2=algo2,
        reg=reg,
        reg2=reg2,
        alpha_label_uplift=alphaT,
        alphaLin=alphaLin,
        beta_fcost=beta_fcost,
        normalize=normalize,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        numIterBCD=numIterBCD,
        perf_reference=target.Y,
        verbose=verbose,
    )

    # ---- block 2: target(train) -> source(all) ----
    y_target2 = y_target_train.copy()
    y_source2 = y_source.copy()
    y_source2[l_source_test] = -1
    M_lin_2 = compute_cost_matrix(yt=y_source2, ys=y_target2)
    M_lin_2 = _mask_unknown_labels(M_lin_2, np.asarray(l_source_test), axis=1)

    clf_source = _run_bcd_block(
        x_anchor_train=x_target_train,
        y_anchor_train=y_target_train,
        x_full=x_source,
        y_full_train_known=y_source_train,
        l_full_train=l_source_train,
        l_full_test=l_source_test,
        clf_full=clf_source,
        M_lin=M_lin_2,
        algo1=algo1,
        algo2=algo2,
        reg=reg,
        reg2=reg2,
        alpha_label_uplift=alphaS,
        alphaLin=alphaLin,
        beta_fcost=beta_fcost,
        normalize=normalize,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        numIterBCD=numIterBCD,
        perf_reference=source.Y,
        verbose=verbose,
    )

    ypred_target = clf_target.predict(x_target[l_target_test, :], verbose=0).ravel()
    ypred_source = clf_source.predict(x_source[l_source_test, :], verbose=0).ravel()

    perf_pure_source = continuous_accuracy(ypred_source, source.loc[l_source_test, "Y"])
    perf_pure_target = continuous_accuracy(ypred_target, target.loc[l_target_test, "Y"])

    zt_test = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()
    zs_test = clf_source.predict(
        source_test.loc[:, xcolumns(source_test)], verbose=0
    ).ravel()

    perf_test_source = continuous_accuracy(zs_test, source_test.Y)
    perf_test_target = continuous_accuracy(zt_test, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
