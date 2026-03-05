def general_coot(
    source,
    target,
    source_test,
    target_test,

    # ====== TASK ======
    task="continuous",                  # "continuous" | "discrete"
    supervision="unsupervised",          # "unsupervised" | "semisupervised" | "partial"

    # ====== SPLITS ======
    prop_source=0.1,
    prop_target=0.1,
    random_state=None,

    # ====== COOT ======
    niter=100,
    algo="sinkhorn",
    algo2="emd",
    reg=100,
    reg2=None,
    verbose=False,

    # ====== COST ======
    regression_metric=None,              # None → comp_regression()
    discrete_cost_v=10000,

    # ====== CLASSIFIER ======
    batch_size=20,
    epochs=20,
    clf_source_kwargs=None,
    clf_target_kwargs=None,
):
    """
    Unified COOT pipeline for continuous / discrete tasks with all hyperparameters exposed.
    """

    # =========================
    # 1. Features
    # =========================
    Xs = source.loc[:, xcolumns(source)].values
    Xt = target.loc[:, xcolumns(target)].values

    # =========================
    # 2. Labels + encoders
    # =========================
    if task == "continuous":
        Ys = source.Y.values.astype(float)
        Yt = target.Y.values.astype(float)

        accuracy = continuous_accuracy
        cost_metric = regression_metric or comp_regression()

        def encode(y):
            return y

        def decode(y):
            return y

        build_clf = continuous_classifier
        build_clfs = continuous_classifiers

    else:  # DISCRETE
        Ys = source.Z.values.astype(int)
        Yt = target.Z.values.astype(int)

        classes = np.union1d(np.unique(Ys), np.unique(Yt))
        nClass = len(classes)

        accuracy = discrete_accuracy
        cost_metric = comp_(discrete_cost_v)

        def encode(z):
            return to_categorical(z, num_classes=nClass)

        def decode(z):
            return np.argmax(z, axis=1)

        build_clf = lambda t: discrete_classifier(
            t, **(clf_target_kwargs or {}), nClass=nClass
        )
        build_clfs = lambda s, t: discrete_classifiers(
            s, t, **(clf_source_kwargs or {})
        )

    # =========================
    # 3. Supervision masks
    # =========================
    Ys_train = Ys.copy()
    Yt_train = Yt.copy()

    ls_test = np.array([], dtype=int)
    lt_test = np.array([], dtype=int)

    if supervision == "partial": ### pour donnees reelles ca doit etre donné en entree
        ls_train, ls_test = train_test_split(
            np.arange(len(Ys)),
            train_size=prop_source,
            random_state=random_state,
        )
        lt_train, lt_test = train_test_split(
            np.arange(len(Yt)),
            train_size=prop_target,
            random_state=random_state,
        )

        Ys_train[ls_test] = np.nan if task == "continuous" else -1
        Yt_train[lt_test] = np.nan if task == "continuous" else -1

    elif supervision == "semisupervised":
        lt_train, lt_test = train_test_split(
            np.arange(len(Yt)),
            train_size=prop_target,
            random_state=random_state,
        )
        Yt_train[lt_test] = np.nan if task == "continuous" else -1

    else:  # UNSUPERVISED
        lt_test = np.arange(len(Yt))

    # =========================
    # 4. Cost matrix
    # =========================
    def cost_matrix(y1, y2):
        return ot.dist(
            y1.reshape(-1, 1),
            y2.reshape(-1, 1),
            metric=cost_metric,
        )

    C_lin = cost_matrix(Ys, Yt_train)

    # =========================
    # 5. COOT
    # =========================
    Ts, _, _ = cot_numpy(
        X1=Xs,
        X2=Xt,
        C_lin=C_lin,
        niter=niter,
        algo=algo,
        algo2=algo2,
        reg=reg,
        reg2=reg2,
        verbose=verbose,
    )

    Yt_pred = len(Yt) * Ts.T @ encode(Ys)

    Yt_pred_decoded = decode(Yt_pred)

    # =========================
    # 6. Pure transport score
    # =========================
    perf_pure_source = 0.0
    perf_pure_target = accuracy(
        Yt_pred_decoded[lt_test],
        Yt[lt_test],
    )

    # =========================
    # 7. Classifier
    # =========================
    clf = build_clf(target)

    clf.fit(
        Xt,
        Yt_pred,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )

    Xt_test = target_test.loc[:, xcolumns(target_test)].values
    Yt_test_pred = clf.predict(Xt_test, verbose=0)

    Yt_test_pred = decode(Yt_test_pred)

    perf_test_source = 0.0
    perf_test_target = accuracy(
        Yt_test_pred,
        target_test.Y if task == "continuous" else target_test.Z,
    )

    return (
        perf_pure_source,
        perf_pure_target,
        perf_test_source,
        perf_test_target,
    )







