import tensorflow as tf

class WeightedAdd(tf.keras.layers.Add):
    """Acts like Add, but does (1–α)*x + α*h instead of x+h."""
    def __init__(self, alpha=0.45, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        x, h = inputs
        return (1.0 - self.alpha) * x + self.alpha * h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha})
        return cfg
