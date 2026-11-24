"""Export a Keras model to TFLite (float32 / dynamic-range / full int8) and benchmark it."""
import time
import numpy as np
import tensorflow as tf


def to_tflite(model, path, quant=None, rep_data=None):
    """quant: None (float32), "dynamic" (int8 weights) or "int8" (full int8, needs rep_data [N,128,9])."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant in ("dynamic", "int8"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if quant == "int8":
        assert rep_data is not None, "int8 needs rep_data"
        conv.representative_dataset = lambda: ([x[None].astype("float32")] for x in rep_data)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int8
    elif quant is not None and quant != "dynamic":
        raise ValueError(quant)
    blob = conv.convert()
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def predict_tflite(path, X, timed=False):
    """Run X [N,128,9] one window at a time. Returns probs [N,C] (and mean ms/window if timed)."""
    it = tf.lite.Interpreter(model_path=path, num_threads=1)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out, t = [], 0.0
    for x in X:
        x = x[None].astype("float32")
        if i["dtype"] == np.int8:
            s, z = i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        it.set_tensor(i["index"], x)
        it.invoke()
        y = it.get_tensor(o["index"])[0]
        t += time.perf_counter() - t0
        if o["dtype"] == np.int8:
            s, z = o["quantization"]
            y = (y.astype("float32") - z) * s
        out.append(y)
    out = np.array(out)
    return (out, 1000 * t / len(X)) if timed else out


if __name__ == "__main__":
    import os, tempfile
    m = tf.keras.Sequential([tf.keras.layers.Input((128, 9)), tf.keras.layers.Conv1D(4, 3),
                             tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(6, activation="softmax")])
    X = np.random.randn(20, 128, 9).astype("float32")
    d = tempfile.mkdtemp()
    for q in (None, "dynamic", "int8"):
        p = os.path.join(d, f"{q}.tflite")
        to_tflite(m, p, q, X)
        assert predict_tflite(p, X[:3]).shape == (3, 6)
    print("ok")
