def selection_only(ds, stats, frequency: str, horizon: int):
    normalized = apply_normalizer(ds, stats, frequency)
    X = flatten_features(frequency, normalized)
    y = ds.y[horizon]
    train_mask = ds.split == 'train'
    val_mask = ds.split == 'val'
    test_mask = ds.split == 'test'
    best_alpha, best_val_mse, best_model = (None, np.inf, None)
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha)
        model.fit(X[train_mask], y[train_mask])
        val_pred = model.predict(X[val_mask])
        val_mse = float(np.mean((val_pred - y[val_mask]) ** 2))
        if val_mse < best_val_mse:
            best_alpha, best_val_mse, best_model = (alpha, val_mse, model)
    return (best_alpha, best_val_mse, best_model, X)
