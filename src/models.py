"""Owner: Yash. Deep model uses Keras (tensorflow), not PyTorch."""
from sklearn.ensemble import RandomForestClassifier


def build_rf():
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def build_cnn(input_shape=(128, 9), n_classes=6):
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(64, 5, activation="relu"),
        layers.Conv1D(64, 5, activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_cnn_gravity(input_shape=(128, 9), n_angles=3, n_classes=6):
    """build_cnn plus a second input: window-mean gravity angles, concatenated before the dense layer."""
    from tensorflow import keras
    from tensorflow.keras import layers
    raw, ang = keras.Input(shape=input_shape), keras.Input(shape=(n_angles,))
    x = layers.Conv1D(64, 5, activation="relu")(raw)
    x = layers.Conv1D(64, 5, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(layers.MaxPooling1D(2)(x))
    x = layers.Dense(100, activation="relu")(layers.Concatenate()([x, ang]))
    m = keras.Model([raw, ang], layers.Dense(n_classes, activation="softmax")(x))
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
