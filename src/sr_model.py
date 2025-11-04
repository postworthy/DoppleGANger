
import tensorflow as tf
from simple_weighted_skip import simple_weighted_skip
#from onnx_upsampler import Upsampler_remote
#from inference import AEINETSwapper

class FaceSuperResolutionModel(tf.keras.Model):
    """
    Lightweight CNN for 2× face super-resolution (256×256 → 512×512),
    with a custom train_step that uses an external ONNX upsampler to generate HR targets.
    """
    def __init__(self, name=None, upsampler=None, face_swapper=None):
        super(FaceSuperResolutionModel, self).__init__(name=name)
        # Feature extraction
        self.conv1 = tf.keras.layers.Conv2D(
            64, (5, 5), padding='same', activation='relu', name='feature_extract'
        )
        # Channel shrinking
        self.conv2 = tf.keras.layers.Conv2D(
            32, (1, 1), padding='same', activation='relu', name='shrink'
        )
        # Mapping layers
        self.conv_blocks = [
            tf.keras.layers.Conv2D(
                32, (3, 3), padding='same', activation='relu',
                name=f'map_{i}'
            ) for i in range(4)
        ]
        # Final expanding: output 3*(2^2)=12 channels for pixel shuffle
        self.conv_expand = tf.keras.layers.Conv2D(
            3 * 4, (3, 3), padding='same', activation='relu', name='expand'
        )
        # External ONNX upsampler to generate HR targets
        self.upsampler = upsampler
        self.face_swapper = face_swapper
    
    def call(self, inputs, training=False):
        """
        Forward pass of the SR network:
        inputs: [B,256,256,3]
        returns: [B,512,512,3]
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.conv_expand(x)
        x = tf.nn.depth_to_space(x, block_size=2)

        # Residual skip connection
        skip = tf.image.resize(inputs, [512, 512], method='bilinear')
        skip = tf.cast(skip, x.dtype)
        return simple_weighted_skip([x, skip], 0.45, "SR")
        #return x + skip

    def upsample_fn(self, lr):
        """
        Use the external ONNX upsampler to generate a 512×512 HR target.
        lr: [B,256,256,3], dtype float (e.g., float16 or float32)
        returns: [B,512,512,3], same dtype
        """
        # Convert to float32 for ONNX if needed
        x = tf.cast(lr, tf.float32)
        # If ONNX expects BGR, convert here; otherwise skip reverse
        x = tf.reverse(x, axis=[-1])
        # Perform ONNX upsampling
        up = self.call_upsampler(x, training=False)  # [B,512,512,3], float32
        # Clip to [0,1]
        up = tf.clip_by_value(up, 0.0, 1.0)
        # Reverse channels back
        up = tf.reverse(up, axis=[-1])
        # Match model dtype
        return tf.cast(up, lr.dtype)

    @tf.function(jit_compile=False)
    def call_upsampler(self, x, training=True):
        # ONNX upsampling call: returns float32
        if self.upsampler:
            return self.upsampler.onnx_upsample(x, False)
        else:
            raise Exception("self.upsampler must be set during training")

    @tf.function
    def train_step_old(self, data):
        """
        Training step using ONNX upsampler for HR target.
        """
        # Unpack data tuple; ignore img_batch/embed_batch
        (_, _), (lr, _) = data

        # Generate HR using external upsampler
        hr = self.upsample_fn(lr)  # may not exactly match spatial size

        with tf.GradientTape() as tape:
            # Predict SR output
            sr = self(lr, training=True)  # [B, H, W, 3], H/W likely 512

            # Ensure hr matches sr spatial resolution
            hr = tf.image.resize(hr, size=tf.shape(sr)[1:3], method='bilinear')

            # Cast to float32 for stable loss
            hr32 = tf.cast(hr, tf.float32)
            sr32 = tf.cast(sr, tf.float32)

            # Compute L1 loss
            loss = tf.reduce_mean(tf.abs(hr32 - sr32))
            # Scale for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Compute and apply gradients
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(grads)
        # Optional: clip gradients
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics (e.g., track MAE)
        self.compiled_metrics.update_state(hr32, sr32)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results
        
    @tf.function
    def train_step(self, data):
        """
        Training step using ONNX upsampler for HR target.
        """
        # Unpack data tuple; ignore img_batch/embed_batch
        (input_batch, embed_batch), (lr, _) = data

        with tf.GradientTape() as tape:
            if tf.random.uniform(()) > 0.5:
                preds, _ = self.face_swapper([input_batch, embed_batch], training=False)
                sr = self(preds, training=True)
            else:
                sr = self(lr, training=True)

            if tf.random.uniform(()) > 0.5:
                hr = tf.image.resize(lr, size=tf.shape(sr)[1:3], method='bilinear')
            else:
                hr = tf.image.resize(lr, size=tf.shape(sr)[1:3], method='area')

            # Cast to float32 for stable loss
            hr32 = tf.cast(hr, tf.float32)
            sr32 = tf.cast(sr, tf.float32)

            # Compute L1 loss
            loss = tf.reduce_mean(tf.abs(hr32 - sr32))
            # Scale for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Compute and apply gradients
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(grads)
        # Optional: clip gradients
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics (e.g., track MAE)
        self.compiled_metrics.update_state(hr32, sr32)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results
    
    @tf.function
    def test_step(self, data):
        # ((img_batch, embed_batch), (lr, _))
        (_, _), (lr, _) = data
        hr = self.upsample_fn(lr)

        # Forward pass
        sr = self(lr, training=False)
        hr_resized = tf.image.resize(hr, tf.shape(sr)[1:3], method='bilinear')

        # Metrics only
        sr32 = tf.cast(sr, tf.float32)
        hr32 = tf.cast(hr_resized, tf.float32)
        self.compiled_metrics.update_state(hr32, sr32)
        results = {m.name: m.result() for m in self.metrics}
        return results
    
    def wireable(self,
                 input_shape=(256, 256, 3),
                 name: str | None = None) -> tf.keras.Model:
        """
        Return a tf.keras.Model that can be plugged into a larger Functional
        graph.  The wrapper shares this instance's layers/variables, so any
        training done through the outer model still updates the SR network.
        """
        if hasattr(self, "_wireable"):          # build once, then cache
            return self._wireable

        # 1. Create a symbolic input
        x = tf.keras.Input(shape=input_shape, name="sr_input")

        # 2. Route it through *this* model (variables are created on first call)
        y = self(x, training=False)             # training flag will be passed
                                                # by the outer model later

        # 3. Wrap as a Functional sub‑model
        self._wireable = tf.keras.Model(x, y,
                                        name=name or f"{self.name}_wireable")
        return self._wireable