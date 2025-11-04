import tensorflow as tf
from tensorflow.keras.layers import Lambda

def simple_weighted_skip(tensor_inputs, skip_alpha=0.45, name="1"):
    weighted_skip = Lambda(
        lambda tensors: tensors[0] * (1.0 - skip_alpha)
                        + tensors[1] * skip_alpha,
        name=f"scaled_skip_{name}"
    )(tensor_inputs)
    weighted_skip = Lambda(
        lambda t: tf.clip_by_value(t, 0.0, 1.0),
        name=f"clamp_{name}"
    )(weighted_skip)
    return weighted_skip