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
