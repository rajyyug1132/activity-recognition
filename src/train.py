"""Owner: Yash."""


def fit(model, X_tr, y_tr, X_val, y_val, cfg):
    """Returns (model, history). history is None for sklearn models."""
    if not hasattr(model, "compile"):                     # sklearn: no epochs
        return model.fit(X_tr, y_tr), None
    from tensorflow import keras
    keras.utils.set_random_seed(cfg["seed"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    h = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=30,
                  batch_size=64, callbacks=[stop], verbose=2)
    return model, h.history


def fine_tune(model, X_tr, y_tr, X_val, y_val, cfg, n_trainable=2):
    """Transfer learning: copy the UCI model, freeze all but the last `n_trainable` layers,
    recompile with Adam(1e-4) and train on our data. The original model is left untouched,
    so it can still be scored as the "no fine-tuning" baseline. Returns (model, history)."""
    from tensorflow import keras
    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    for layer in m.layers[:-n_trainable]:
        layer.trainable = False
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return fit(m, X_tr, y_tr, X_val, y_val, cfg)
