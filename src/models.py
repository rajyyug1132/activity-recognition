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


def build_lstm(input_shape=(128, 9), n_classes=6):
    """Comparison model for the CNN: 2 stacked LSTMs on raw windows."""
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dropout(0.5),
        layers.Dense(100, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
