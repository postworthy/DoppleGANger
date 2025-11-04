from aei_net import AAD_ResBlk
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation, Input
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import load_model

class SuperResModel(tf.keras.Model):
    def __init__(self, generator, discriminator, upscale_factor=2, refinement_filters=64, 
                 lambda_recon=100.0, lambda_adv=0.25, lambda_id=10.0,
                 num_blocks=4,
                 name='superres_model'):
            """
            generator: Your AEI-Net model
              - Call signature: Y_pred, z_attr = generator([img_batch, embed_batch])
            discriminator: MinimalPatchGAN
            lambda_recon, lambda_adv: weighting factors for reconstruction vs. adversarial
            """
            super().__init__(name=name)

            self.generator = generator
            self.discriminator = discriminator

            self.arcface = load_model("ArcFace-Res50.h5", compile=False)
            #self.arcface(tf.zeros((1, 112, 112, 3), dtype=tf.float32))  # "Build" it.
            
            self.lambda_recon = lambda_recon
            self.lambda_adv = lambda_adv
            self.lambda_id = lambda_id

            # Define two inputs: the generator's image output and the identity embedding.
            img_input = Input(shape=(64, 64, 3), name="gen_image_input")
            # Assuming the identity embedding is 512-dimensional.
            embed_input = Input(shape=(512,), name="id_embedding_input")

            # 1. Map the generator's output into a feature space.
            refined_features = Conv2D(
                filters=refinement_filters,
                kernel_size=3,
                padding='same',
                activation='relu'
            )(img_input)

            # 2. Compute an auxiliary attribute map from the refined features.
            z_attr = Conv2D(
                filters=refinement_filters,
                kernel_size=3,
                padding='same',
                activation='relu'
            )(refined_features)

            # 3. Apply the AAD_ResBlk on the refined features BEFORE upsampling.
            #    Note: the original AAD_ResBlk usage typically precedes UpSampling2D.
            aad_features = AAD_ResBlk(
                cin=refinement_filters,
                cout=refinement_filters,
                c_attr=refinement_filters,
                c_id=512,
                num_blocks=num_blocks
            )([refined_features, z_attr, embed_input])

            # 4. Upsample the output of the AAD block.
            upsampled_output = UpSampling2D(size=upscale_factor, interpolation='bilinear')(aad_features)

            # 5. Final convolution and activation to produce the 3‑channel RGB output.
            final_output = Conv2D(
                filters=3,
                kernel_size=3,
                padding='same'
            )(upsampled_output)
            y_final = Activation('tanh', dtype='float32')(final_output)

            # Build the super-resolution branch with two inputs.
            self.superres = Model(
                name="super_resolution_model",
                inputs=[img_input, embed_input],
                outputs=[y_final]
            )
            
            
            # We can define metrics to track
            self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
            self.recon_loss_metric = tf.keras.metrics.Mean(name="recon_loss")
            self.id_loss_metric = tf.keras.metrics.Mean(name="id_loss")
            self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
            self.adv_loss_metric = tf.keras.metrics.Mean(name="adv_loss")

    def extract_embed(self, y_pred):
        try:
            resized_batch = tf.image.resize(y_pred, size=(112, 112))
            embed = self.arcface(resized_batch)
            return embed
        except Exception as e:
            print(f"Error during embedding extraction: {e}")
            batch_size = y_pred.shape[0]
            embed_shape = (batch_size, 512)  # Assuming ArcFace outputs 512-dimensional embeddings
            noise = tf.random.normal(embed_shape, mean=0.0, stddev=1.0)
            return noise

    def id_loss_default(self, y_target, y_pred):
        embed_true = self.extract_embed(y_target)
        embed_pred = self.extract_embed(y_pred)

        embed_true_normalized = tf.nn.l2_normalize(embed_true, axis=1)
        embed_pred_normalized = tf.nn.l2_normalize(embed_pred, axis=1)
        cosine_similarity = tf.reduce_sum(embed_true_normalized * embed_pred_normalized, axis=1)
        return tf.reduce_mean(1 - cosine_similarity)
            
    def compile(self, 
                g_optimizer=AdamW(learning_rate=4e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                d_optimizer=AdamW(learning_rate=4e-5, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                bce_loss_fn=None,
                recon_loss_fn=None,
                id_loss_fn=None):
        """
        g_optimizer, d_optimizer: tf.keras optimizers for G and D
        bce_loss_fn: function that does BCE, e.g. tf.keras.losses.BinaryCrossentropy(from_logits=True)
        recon_loss_fn: e.g. L1 or L2
        """
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        if bce_loss_fn is None:
            self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.bce_loss_fn = bce_loss_fn

        if recon_loss_fn is None:
            # default: L1
            self.recon_loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred))
        else:
            self.recon_loss_fn = recon_loss_fn
            
        if id_loss_fn is None:
            self.id_loss_fn = self.id_loss_default
        else:
            self.id_loss_fn = id_loss_fn
            
    @property
    def metrics(self):
        """
        Return a list of the metrics to reset/update.
        This ensures Keras calls .reset_states() on each epoch.
        """
        return [self.recon_loss_metric, self.id_loss_metric]
    
    def d_loss_fn(self, real_pred, fake_pred):
        # real_pred, fake_pred: PatchGAN outputs
        real_loss = self.bce_loss_fn(tf.ones_like(real_pred), real_pred)
        fake_loss = self.bce_loss_fn(tf.zeros_like(fake_pred), fake_pred)
        return 0.5 * (real_loss + fake_loss)

    def g_adv_loss_fn(self, fake_pred):
        # generator wants D(fake) -> 1
        return self.bce_loss_fn(tf.ones_like(fake_pred), fake_pred)

    def train_step(self, data):
        
        (img_batch, embed_batch), (y_target, y_target_x2) = data

        y_pred, _ = self.generator([img_batch, embed_batch], training=False)
        
        with tf.GradientTape() as d_tape:
            superres_pred = self.superres([y_pred, embed_batch], training=True)
            # D outputs
            d_real = self.discriminator(y_target_x2, training=True)
            d_fake = self.discriminator(superres_pred, training=True)

            d_loss = self.d_loss_fn(d_real, d_fake)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        
        with tf.GradientTape() as superres_tape:
            #superres_pred = self.superres(y_pred, training=True)
            superres_pred = self.superres([y_pred, embed_batch], training=True)
            d_fake_for_g = self.discriminator(superres_pred, training=False)

            adv_loss = self.g_adv_loss_fn(d_fake_for_g)
            # Compute reconstruction loss at 2x resolution
            recon_loss = self.recon_loss_fn(y_target_x2, superres_pred)
            
            #Compute id loss
            id_loss = self.id_loss_fn(y_target_x2, superres_pred)
            
            adv_loss = tf.cast(adv_loss, tf.float32)
            recon_loss = tf.cast(recon_loss, tf.float32)
            id_loss = tf.cast(id_loss, tf.float32)

            g_loss = (self.lambda_adv * adv_loss) + (self.lambda_recon * recon_loss) + (self.lambda_id * id_loss)

        # Compute gradients for superres only
        superres_grads = superres_tape.gradient(g_loss, self.superres.trainable_variables)
        self.g_optimizer.apply_gradients(zip(superres_grads, self.superres.trainable_variables))

        #Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.recon_loss_metric.update_state(recon_loss)
        self.id_loss_metric.update_state(id_loss)
        self.adv_loss_metric.update_state(adv_loss)

        # Return metrics specific to superres
        return {
            "d_loss":self.d_loss_metric.result(),
            "g_loss":self.g_loss_metric.result(),
            "recon_loss": self.recon_loss_metric.result(),
            "adv_loss": self.adv_loss_metric.result(),
            "id_loss": self.id_loss_metric.result(),
        }

        # Normal GAN training logic if not in superres mode
        return super().train_step(data)
    
    def test_step(self, data):
        """
        If you provide validation_data=... to model.fit, 
        Keras will call test_step on each batch of val data.
        We'll measure just reconstruction loss as 'val_loss' for example.
        """
        (img_batch, embed_batch),  (y_target, y_target_x2) = data

        superres_pred = self.superres([y_target, embed_batch], training=False)
        # compute identity loss
        id_loss = self.id_loss_fn(y_target_x2, y_target)

        return {"val_loss": id_loss}
    
    def call(self, inputs, training=None, mask=None):
        """
        This method is used when you do `aei_gan_model([imgs, embeds])`.
        We'll just forward to the generator and return its output.
        """
        # inputs should be `[img_batch, embed_batch]` as in your code.
        # AEI-Net typically returns `(Y_pred, z_attr)`.
        y_pred, z_attr = self.generator(inputs, training=training)
        superres_pred = self.superres([y_pred, inputs[1]], training=training)
        # Return the generated images for inference
        return superres_pred, y_pred
