import tensorflow as tf
from tensorflow.keras import layers, Model

class MinimalPatchGAN(Model):
    """
    A simple single-scale PatchGAN:
      - A few convolution layers that downsample by stride=2.
      - Each conv is followed by LeakyReLU (and optional batch norm).
      - Ends with a 1-channel conv for patch-level real/fake output.
    """
    def __init__(self, ndf=64, num_layers=3, name='MinimalPatchGAN'):
        """
        ndf: base number of discriminator filters
        num_layers: how many conv layers (excluding final 1-channel conv)
        """
        super().__init__(name=name)
        self.ndf = ndf
        self.num_layers = num_layers

        # We'll store the layers in a list
        self.convs = []

        # 1) First layer: Conv + LeakyReLU
        #   Typically stride=2 to reduce spatial size
        self.convs.append(
            layers.Conv2D(filters=ndf, kernel_size=4, strides=2, padding='same', use_bias=True)
        )
        self.convs.append(layers.LeakyReLU(alpha=0.2))

        # 2) Middle layers
        nf_prev = ndf
        for n in range(1, num_layers):
            nf_cur = min(nf_prev * 2, 512)
            self.convs.append(
                layers.Conv2D(filters=nf_cur, kernel_size=4, strides=2, 
                              padding='same', use_bias=False)
            )
            self.convs.append(layers.BatchNormalization())  # optionally remove if you want simpler
            self.convs.append(layers.LeakyReLU(alpha=0.2))
            nf_prev = nf_cur

        # 3) Final layer: 1-channel conv, stride=1 => patch map
        self.convs.append(
            layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same')
        )

    def call(self, x, training=True):
        """
        x: input tensor, shape [batch, H, W, C]
        Returns: patch map shape [batch, H/2^N, W/2^N, 1]
        """
        out = x
        for layer in self.convs:
            # If it's a BatchNormalization, pass `training=training`
            if isinstance(layer, layers.BatchNormalization):
                out = layer(out, training=training)
            else:
                out = layer(out)
        return out
