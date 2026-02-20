import numpy as np
import ot
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical

from ..utils import (
    xcolumns,
    continuous_accuracy,
    discrete_accuracy,
    continuous_classifiers,
    discrete_classifiers,
)
from ..coot import init_matrix_np
from ..losses import loss_crossentropy2


def jdcoot(
    source,
    target,
    source_test=None,
    target_test=None,
    *,
    task="continuous",          # "continuous" | "discrete"
    supervision="partial",      # "partial" | "semi" | "unsupervised"
    prop_source=0.1,
    prop_target=0.1,
    alpha=1.0,
    algo_s="sinkhorn",
    algo_v="emd",
    reg_s=100,
    reg_v=0,
    numIterBCD=100,
    nb_epoch=20,
    batch_size=20,
    verbose=True,
):

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    nA, dA = x_source.shape
    nB, dB = x_target.shape

    vA = np.ones(dA) / dA
    vB = np.ones(dB) / dB
    wA = np.ones(nA) / nA
    wB = np.ones(nB) / nB

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    if task == "continuous":
        y_source = source.Y.values[:, None]
        y_target = target.Y.values[:, None]
        loss_fn = lambda A, B: ot.dist(A, B, metric="sqeuclidean")
        accuracy = continuous_accuracy
        clf_source, clf_target = continuous_classifiers(source, target)

    else:
        classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
        nClass = len(classes)

        def one_hot(y):
            return to_categorical(y, num_classes=nClass)

        def one_cold(z):
            return np.argmax(z, axis=1)

        y_source = one_hot(source.Z).astype(np.float64)
        y_target = one_hot(target.Z).astype(np.float64)

        loss_fn = loss_crossentropy2
        accuracy = discrete_accuracy
        clf_source, clf_target = discrete_classifiers(
            source, target, "sigmoid", "sigmoid"
        )

    # --------------------------------------------------
    # Train / test splits
    # --------------------------------------------------
    if supervision in ("partial", "semi"): ### pour donnees reelles ca doit etre donné en entree
        if supervision == "partial":
            ls_tr, ls_te = train_test_split(
                np.arange(nA), train_size=prop_source
            )
        else:
            ls_tr, ls_te = [], np.arange(nA)

        lt_tr, lt_te = train_test_split(
            np.arange(nB), train_size=prop_target
        )
    else:
        ls_tr, ls_te = [], np.arange(nA)
        lt_tr, lt_te = [], np.arange(nB)

    # --------------------------------------------------
    # COOT init
    # --------------------------------------------------
    C_s, h1_s, h2_s = init_matrix_np(x_source, x_target, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_source.T, x_target.T, wA, wB)

    Gs = np.ones((nA, nB)) / (nA * nB)
    Gv = np.ones((dA, dB)) / (dA * dB)

    # --------------------------------------------------
    # Initial classifiers
    # --------------------------------------------------
    if supervision != "unsupervised":
        if len(ls_tr) > 0:
            clf_source.fit(
                x_source[ls_tr], y_source[ls_tr],
                batch_size=batch_size, epochs=nb_epoch, verbose=0
            )
        clf_target.fit(
            x_target[lt_tr], y_target[lt_tr],
            batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

    y_source_pred = clf_source.predict(x_source, verbose=0)
    y_target_pred = clf_target.predict(x_target, verbose=0)

    if len(ls_tr) > 0:
        y_source_pred[ls_tr] = y_source[ls_tr]
    if len(lt_tr) > 0:
        y_target_pred[lt_tr] = y_target[lt_tr]

    fcost = loss_fn(y_source_pred, y_target_pred)
    cost = np.inf

    # --------------------------------------------------
    # BCD loop
    # --------------------------------------------------
    for k in range(numIterBCD):
        Gsold, Gvold = Gs.copy(), Gv.copy()
        costold = cost

        Ms = (C_s - h1_s @ Gv @ h2_s.T) + alpha * fcost
        Gs = ot.sinkhorn(wA, wB, Ms, reg_s) if algo_s == "sinkhorn" else ot.emd(wA, wB, Ms)

        Mv = C_v - h1_v @ Gs @ h2_v.T
        Gv = ot.sinkhorn(vA, vB, Mv, reg_v) if algo_v == "sinkhorn" else ot.emd(vA, vB, Mv)

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        # label propagation
        if supervision != "unsupervised":
            y_source_hat = nA * Gs @ y_target_pred
            y_target_hat = nB * Gs.T @ y_source_pred

            if len(ls_tr) > 0:
                y_source_hat[ls_tr] = y_source[ls_tr]
            if len(lt_tr) > 0:
                y_target_hat[lt_tr] = y_target[lt_tr]

            clf_source.fit(
                x_source, y_source_hat,
                batch_size=batch_size, epochs=nb_epoch, verbose=0
            )
            clf_target.fit(
                x_target, y_target_hat,
                batch_size=batch_size, epochs=nb_epoch, verbose=0
            )

            y_source_pred = clf_source.predict(x_source, verbose=0)
            y_target_pred = clf_target.predict(x_target, verbose=0)

            if len(ls_tr) > 0:
                y_source_pred[ls_tr] = y_source[ls_tr]
            if len(lt_tr) > 0:
                y_target_pred[lt_tr] = y_target[lt_tr]

            fcost = loss_fn(y_source_pred, y_target_pred)

        if verbose:
            print(f"Iter {k:3d} | Δ={delta:.3e} | cost={cost:.3e}")

        if delta < 1e-16 or abs(cost - costold) < 1e-7:
            break

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    perf_source = 0.0
    perf_target = accuracy(
        y_target_pred[lt_te].ravel() if task == "continuous"
        else np.argmax(y_target_pred[lt_te], axis=1),
        target.Y.values[lt_te] if task == "continuous" else target.Z.values[lt_te]
    )

    perf_test_source = perf_test_target = None
    if source_test is not None:
        xs = source_test.loc[:, xcolumns(source_test)].values
        xt = target_test.loc[:, xcolumns(target_test)].values

        ys = clf_source.predict(xs, verbose=0).ravel()
        yt = clf_target.predict(xt, verbose=0).ravel()

        perf_test_source = accuracy(ys, source_test.Y)
        perf_test_target = accuracy(yt, target_test.Y)

    return perf_source, perf_target, perf_test_source, perf_test_target