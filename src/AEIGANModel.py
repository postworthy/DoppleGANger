import tensorflow as tf
#from face_analysis import FaceAnalysis
from tensorflow_addons.optimizers import AdamW
from emap import emap
import numpy as np
from tensorflow.keras.models import load_model
import contextlib
import cv2

emap_tf = tf.convert_to_tensor(emap)

class AEIGANModel(tf.keras.Model):
    """
    Custom model to train AEI-Net (generator) + PatchGAN (discriminator)
    using model.fit(...) in a Keras workflow.
    """
    def __init__(self, generator=None, discriminator=None,
                 teacher=None,
                 upsampler=None,
                 lambda_recon=100.0, lambda_adv=0.25, lambda_id=10.0, lambda_upsample=50.0, lambda_sharp=0.0, lambda_color=0.5,
                 freeze=False,
                 has_superres_generator=False,
                 name='aei_gan_model',
                 use_emap=True):
        """
        generator: Your AEI-Net model
          - Call signature: Y_pred, z_attr = generator([img_batch, embed_batch])
        discriminator: MinimalPatchGAN
        lambda_recon, lambda_adv: weighting factors for reconstruction vs. adversarial
        """
        super().__init__(name=name)
        
        self.has_superres_generator = has_superres_generator

        #self.face_analyser = FaceAnalysis()
        #self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        
        self.generator = generator
        self.discriminator = discriminator
        self.teacher = teacher
        self.upsampler = upsampler

        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_id = lambda_id
        self.lambda_upsample = lambda_upsample
        self.lambda_sharp = lambda_sharp
        self.lambda_color = lambda_color
        
        self.arcface = load_model("ArcFace-Res50.h5", compile=False)
        
        # We can define metrics to track
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.recon_loss_metric = tf.keras.metrics.Mean(name="recon_loss")
        self.id_loss_metric = tf.keras.metrics.Mean(name="id_loss")
        self.adv_loss_metric = tf.keras.metrics.Mean(name="adv_loss")
        self.up_loss_metric = tf.keras.metrics.Mean(name="up_loss")
        self.color_loss_metric = tf.keras.metrics.Mean(name="color_loss")
        #self.sharp_loss_metric = tf.keras.metrics.Mean(name="sharp_loss")

        self.self_batch_counter=0
        self.dataset_name = None
        self.use_emap = use_emap
    
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

    def extract_normed_embed(self, y_pred):
        latent = self.extract_embed(y_pred)
        if self.use_emap:
            latent = tf.linalg.matmul(latent, emap_tf)
        latent = tf.math.l2_normalize(latent, axis=1)
        return latent


    def id_loss_default(self, y_target, y_pred):
        embed_true = self.extract_normed_embed(y_target)
        embed_pred = self.extract_normed_embed(y_pred)

        embed_true_normalized = tf.nn.l2_normalize(embed_true, axis=1)
        embed_pred_normalized = tf.nn.l2_normalize(embed_pred, axis=1)
        cosine_similarity = tf.reduce_sum(embed_true_normalized * embed_pred_normalized, axis=1)
        return tf.reduce_mean(1 - cosine_similarity)


    def compile(self, 
                g_optimizer=AdamW(learning_rate=4e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                d_optimizer=AdamW(learning_rate=4e-5, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                bce_loss_fn=None, 
                recon_loss_fn=None,
                id_loss_fn=None,
                run_eagerly=None,
                split_batch_by=None):
        """
        g_optimizer, d_optimizer: tf.keras optimizers for G and D
        bce_loss_fn: function that does BCE, e.g. tf.keras.losses.BinaryCrossentropy(from_logits=True)
        recon_loss_fn: e.g. L1 or L2
        """
        if run_eagerly:
            super().compile(run_eagerly=run_eagerly)
        else:
            super().compile()
        
        self.split_batch_by=split_batch_by

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        # If no BCE or recon passed in, define defaults
        if bce_loss_fn is None:
            #self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.bce_loss_fn = self._safe_bce

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
        return [self.d_loss_metric, 
                self.g_loss_metric,
                self.recon_loss_metric, 
                self.adv_loss_metric,
                self.id_loss_metric,
                self.up_loss_metric,
                self.color_loss_metric
                ]

    def _safe_bce(self, labels, logits, clip=20.0):
        """Binary cross-entropy on clipped float32 logits to avoid NaN/Inf."""
        logits = tf.cast(logits, tf.float32)
        # Clip logits to keep exp/sigmoid stable
        logits = tf.clip_by_value(logits, -clip, clip)
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # Reduce mean in fp32
        return tf.reduce_mean(loss)

    def d_loss_fn(self, real_pred, fake_pred):
        # real_pred, fake_pred: PatchGAN outputs
        real_loss = self.bce_loss_fn(tf.ones_like(real_pred), real_pred)
        fake_loss = self.bce_loss_fn(tf.zeros_like(fake_pred), fake_pred)
        return 0.5 * (real_loss + fake_loss)

    def g_adv_loss_fn(self, fake_pred):
        # generator wants D(fake) -> 1
        return self.bce_loss_fn(tf.ones_like(fake_pred), fake_pred)

    def pre_fit_setup(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.self_batch_counter = 0
        #if self.upsampler and hasattr(self.upsampler, "initialize_upsampler"):
        #    self.upsampler.initialize_upsampler(dataset_name)
        

    def train_step(self, data):
        #A bit complicated but this is for splitting batches when inference calls for tuning, typical training will skip this big if
        # if self.split_batch_by and self.split_batch_by in [2, 4, 8] and len(data) == 2:
        #     if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
        #         (img_batch, embed_batch), (y_target, y_target_x2) = data
        #         img_batch_sub = tf.split(img_batch, num_or_size_splits=self.split_batch_by, axis=0)
        #         embed_batch_sub = tf.split(embed_batch, num_or_size_splits=self.split_batch_by, axis=0)
        #         y_target_sub = tf.split(y_target, num_or_size_splits=self.split_batch_by, axis=0)
        #         y_target_x2_sub = tf.split(y_target_x2, num_or_size_splits=self.split_batch_by, axis=0)
        #         for i in range(self.split_batch_by):
        #             out = self.train_step256x256(((img_batch_sub[i], embed_batch_sub[i]), (y_target_sub[i],y_target_x2_sub[i])))
        #         return out


        #result = self.train_step256x256(data) #

        rand_val = tf.random.uniform([], 0, 1)

        
        def run_train_baseline():
            return self.train_step256x256_baseline(data)
        
        def run_train_step256x256():
            return self.train_step256x256(data)

        result = tf.cond(rand_val < 0.25, run_train_baseline, run_train_step256x256)


        self.self_batch_counter = self.self_batch_counter + 1
        return result

    # def train_step64x64(self, data):
    #     # Check if data is a tuple or list
    #     if not isinstance(data, (tuple, list)):
    #         raise ValueError(f"Expected data to be a tuple or list, got: {type(data)}")

    #     # We can handle 2 types of training
    #     # 1) Complete prepreocessed of shape => (img_batch, embed_batch), (y_target, y_target_x2)
    #     # 2) No preprocessing  of shape => (img_batch, extract_embeds_from)
    #     # If type #2 then we will perform embedding extraction and use self.teacher to get y_target and y_target_x2
    #     if len(data) == 2:
    #         if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
    #             (img_batch, embed_batch), (y_target, y_target_x2) = data
    #             extract_embeds_from = None
    #         else:
    #             img_batch, extract_embeds_from = data
    #             embed_batch = self.extract_normed_embed(extract_embeds_from)
    #             if self.teacher == None:
    #                 raise ValueError("Teacher model can't be none for this dataset shape!")
    #             y_target = self.teacher([img_batch, embed_batch], training=False)
    #             y_target_x2 = None            
    #     else:
    #         raise ValueError(f"Unexpected data structure length: {len(data)}")
        
    #     # 1) Train Discriminator
    #     with tf.GradientTape() as d_tape:
    #         # forward pass generator
    #         y_pred, _ = self.generator([img_batch, embed_batch], training=True)

    #         # D outputs
    #         d_real = self.discriminator(y_target, training=True)
    #         d_fake = self.discriminator(y_pred, training=True)

    #         d_loss = self.d_loss_fn(d_real, d_fake)

    #     d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
    #     self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

    #     # 2) Train Generator
    #     with tf.GradientTape() as g_tape:
    #         # Re-generate since previous pass is "consumed"
            
    #         #This overrides the data in the dataset
    #         #embed_batch = self.extract_embed(y_target) #Not great if augmentation is heavy
            
    #         y_pred, _ = self.generator([img_batch, embed_batch], training=True)
    #         d_fake_for_g = self.discriminator(y_pred, training=False)

    #         adv_loss = self.g_adv_loss_fn(d_fake_for_g)
    #         recon_loss = self.recon_loss_fn(y_target, y_pred)
    #         id_loss = self.id_loss_fn(y_target, y_pred)
            
    #         adv_loss = tf.cast(adv_loss, tf.float32)
    #         recon_loss = tf.cast(recon_loss, tf.float32)
    #         id_loss = tf.cast(id_loss, tf.float32)
            
    #         g_loss = (self.lambda_adv * adv_loss) + (self.lambda_recon * recon_loss) + (self.lambda_id * id_loss)

    #     g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
    #     self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

    #     # Update metrics
    #     self.d_loss_metric.update_state(d_loss)
    #     self.g_loss_metric.update_state(g_loss)
    #     self.recon_loss_metric.update_state(recon_loss)
    #     self.adv_loss_metric.update_state(adv_loss)
    #     self.id_loss_metric.update_state(id_loss)

    #     # Return dictionary for display in logs
    #     return {
    #         "d_loss": self.d_loss_metric.result(),
    #         "g_loss": self.g_loss_metric.result(),
    #         "recon_loss": self.recon_loss_metric.result(),
    #         "adv_loss": self.adv_loss_metric.result(),
    #         "id_loss": self.id_loss_metric.result(),
    #     }

    @tf.function(jit_compile=True)
    def compute_laplacian_sharpness(self, image):
        def rgb_to_grayscale(image):
            # Ensure image is float32 and in the range [0, 1]
            image = tf.cast(image, tf.float32)
            # Use the standard coefficients for converting RGB to grayscale
            weights = tf.constant([0.2989, 0.5870, 0.1140], shape=[1, 1, 1, 3], dtype=tf.float32)
            gray = tf.reduce_sum(image * weights, axis=-1, keepdims=True)
            return gray

        # Convert to grayscale if the image is RGB
        if image.shape[-1] == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image

        # Define a Laplacian kernel to highlight edges
        laplacian_kernel = tf.constant([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=tf.float32)
        laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])
        
        # Convolve the image with the kernel
        laplacian_response = tf.nn.conv2d(gray, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # The variance of the Laplacian response indicates the level of high-frequency details (i.e., sharpness)
        sharpness = tf.math.reduce_variance(laplacian_response, axis=[1, 2, 3])
        return sharpness
    
    @tf.function(jit_compile=True)
    def sharpness_loss_fn(self, real_image, generated_image):
        # Compute sharpness for both the reference (high-quality) and generated images
        real_sharp = self.compute_laplacian_sharpness(real_image)
        gen_sharp = self.compute_laplacian_sharpness(generated_image)
        # For example, using L1 difference
        loss = tf.reduce_mean(tf.abs(real_sharp - gen_sharp))
        return loss
    
    @tf.function(jit_compile=False)
    def upsample_fn_orig(self, y_target):    
        reversed_rgb = tf.reverse(y_target, axis=[-1])
        upsampled = self.call_upsampler(tf.cast(reversed_rgb, tf.float32))
        upsampled = tf.clip_by_value(upsampled, 0, 1) * 255
        upsampled = tf.cast(upsampled, tf.uint8)
        upsampled = tf.reverse(upsampled, axis=[-1])
        resized = tf.image.resize(upsampled, (256, 256), method=tf.image.ResizeMethod.AREA) #JIT breaks with tf.image.ResizeMethod.AREA
        final =  resized / 255.0

        # Uncomment to see the upsampled images in ./output/
        #def save_tensor_as_image(tensor, filename):
        #    tensor_uint8 = tf.cast(tf.clip_by_value(tensor * 255.0, 0, 255), tf.uint8)
        #    images_list = tf.unstack(tensor_uint8, axis=0)
        #    combined_image = tf.concat(images_list, axis=1)
        #    encoded_image = tf.io.encode_jpeg(combined_image)
        #    tf.io.write_file(filename, encoded_image)
        #save_tensor_as_image(y_target, "./output/y_target_output.jpg")
        #save_tensor_as_image(final, "./output/superres_output.jpg")
        
        final = tf.cast(final, tf.float16)
        return final

    @tf.function(jit_compile=False)
    def upsample_fn(self, y):
        # assume y in [0,1] float32
        up = self.upsampler(y, training=True)            # float32
        up = tf.clip_by_value(up, 0.0, 1.0)              # still float32
        up = tf.image.resize(up,
                            size=(256,256),
                            method=tf.image.ResizeMethod.BILINEAR)
        return up  # float32 in [0,1]


    @tf.function(jit_compile=True)
    def match_mean_brightness(self, y_target, y_pred, eps=1e-6):
        """
        Scale `y_pred` so its mean luminance (Y channel) equals that of `y_target`.
        Works on a single image or a batch in [0,1] float32/16.
        """
        src_y = tf.image.rgb_to_yuv(y_target)[..., :1]          # (N,H,W,1)
        tgt_y = tf.image.rgb_to_yuv(y_pred)[..., :1]

        src_mean = tf.reduce_mean(src_y, axis=[1,2,3], keepdims=True)
        tgt_mean = tf.reduce_mean(tgt_y, axis=[1,2,3], keepdims=True)

        src_mean = tf.cast(src_mean, tf.float16)
        tgt_mean = tf.cast(tgt_mean, tf.float16)

        scale = src_mean / (tgt_mean + eps)                   # (N,1,1,1)
        out = tf.clip_by_value(y_pred * scale, 0.0, 1.0)
        return out

    @tf.function(jit_compile=True)
    def color_loss_fn(self, y_target, y_pred, color_lambda=1.0, brightness_lambda=1.0):
        """
        L1 difference on chroma channels in YUV space.
        Keeps the loss focused on colour, not overall brightness.
        """
        # y_target, y_pred are assumed in [0,1] float32/16 with shape (N,H,W,3)
        true_yuv = tf.image.rgb_to_yuv(tf.cast(y_target, tf.float32))
        pred_yuv = tf.image.rgb_to_yuv(tf.cast(y_pred, tf.float32))

        # Slice chroma (U,V) channels only → shape (...,2)
        true_uv = true_yuv[..., 1:]
        pred_uv = pred_yuv[..., 1:]

        color_l1 = tf.reduce_mean(tf.abs(true_uv - pred_uv))
        color_loss = tf.cast(color_l1, tf.float32)

        bright_matched = self.match_mean_brightness(y_target, y_pred)  
        bright_l1 = tf.reduce_mean(tf.abs(bright_matched - y_pred))
        bright_loss = tf.cast(bright_l1, tf.float32)

        return color_lambda * color_loss + brightness_lambda * bright_loss



    @tf.function(jit_compile=False)
    def call_upsampler(self, y_target, training=True):
        return self.upsampler(y_target, training=training)
        #return self.upsampler.get_upsampled_record(self.dataset_name, self.self_batch_counter)

    @tf.function(jit_compile=True)
    def call_generator(self, input, training=True):
        return self.generator(input, training=training)

    # AEIGANModel.py — change the discriminator call to avoid XLA on fp16 logits
    # Old:
    # @tf.function(jit_compile=True)
    # def call_descriminator(self, y, training=True):
    #     return self.discriminator(y, training=training)

    # New: disable jit just for D, and document why
    @tf.function(jit_compile=True)
    def call_descriminator(self, y, training=True):
        # Run the discriminator numerics in float32 for stability
        y = tf.cast(y, tf.float32)
        out = self.discriminator(y, training=training)
        return tf.cast(out, tf.float32)

    def train_step256x256(self, data):
        # Check if data is a tuple or list
        if not isinstance(data, (tuple, list)):
            print(f"Expected data to be a tuple or list, got: {type(data)}")
            raise ValueError(f"Expected data to be a tuple or list, got: {type(data)}")

        # We can handle 2 types of training
        # 1) Complete prepreocessed of shape => (img_batch, embed_batch), (y_target, y_target_x2)
        # 2) No preprocessing  of shape => (img_batch, extract_embeds_from)
        # If type #2 then we will perform embedding extraction and use self.teacher to get y_target and y_target_x2
        if len(data) == 2:
            if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                (img_batch, embed_batch), (y_target, y_target_x2) = data
                extract_embeds_from = None
            else:
                img_batch, extract_embeds_from = data
                embed_batch = self.extract_normed_embed(extract_embeds_from)
                if self.teacher == None:
                    raise ValueError("Teacher model can't be none for this dataset shape!")
                y_target = self.teacher([img_batch, embed_batch], training=False)
                y_target_x2 = None            
        else:
            print(f"Unexpected data structure length: {len(data)}")
            raise ValueError(f"Unexpected data structure length: {len(data)}")


        with tf.GradientTape(persistent=True) as tape:
            # Forward generator
            y_pred_sr, y_pred, _ = self.call_generator([img_batch, embed_batch], training=True)

            y_pred_sr = tf.cast(y_pred_sr, tf.float16)
            y_pred = tf.cast(y_pred, tf.float16)
            y_target = tf.cast(y_target, tf.float16)


            #resize_methods = [
            #    tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            #    tf.image.ResizeMethod.BILINEAR,
            #    tf.image.ResizeMethod.AREA,
            #    tf.image.ResizeMethod.BICUBIC,
            #]
            #num_methods = len(resize_methods)
            #method_idx = tf.random.uniform(
            #    [], minval=0, maxval=num_methods, dtype=tf.int32
            #)
            #def make_branch_fn(method):
            #    def _branch():
            #        resized = tf.image.resize(y_target, (512, 512), method=method)
            #        return tf.cast(resized, y_target.dtype)
            #    return _branch
            #branch_fns = [make_branch_fn(m) for m in resize_methods]
            #y_target_up = tf.switch_case(
            #    branch_index=method_idx,
            #    branch_fns=branch_fns
            #)


            y_target_up = tf.image.resize(y_target, size=(512,512), method=tf.image.ResizeMethod.BILINEAR)
            #y_target_up = tf.image.resize(y_pred, size=(512,512), method=tf.image.ResizeMethod.BILINEAR) #I want to predict upsampling of the output only with this one
            y_target_up = tf.cast(y_target_up, tf.float16)

            color_loss = (0.55) * self.color_loss_fn(y_target, y_pred) + (0.45) * self.color_loss_fn(y_target_up, y_pred_sr, color_lambda=0.0, brightness_lambda=2.0)

            # Target upsampled data
            if self.upsampler:
                y_target_up = self.upsample_fn(y_target)
                y_target_up = tf.cast(y_target_up, tf.float16)
                
                #y_pred_up = self.upsample_fn(y_pred)
                #def save_tensor_as_image(tensor, filename):
                #    tensor_uint8 = tf.cast(tf.clip_by_value(tensor * 255.0, 0, 255), tf.uint8)
                #    images_list = tf.unstack(tensor_uint8, axis=0)
                #    combined_image = tf.concat(images_list, axis=1)
                #    encoded_image = tf.io.encode_jpeg(combined_image)
                #    tf.io.write_file(filename, encoded_image)
                #save_tensor_as_image(y_pred_up, "./output/output.jpg")
                y_target_up.set_shape((None, y_target.shape[1], y_target.shape[2], 3))  
                upsample_loss = self.recon_loss_fn(y_target_up, y_pred_sr)
            else:
                #upsample_loss = 0
                upsample_loss = self.recon_loss_fn(y_target_up, y_pred_sr)


            # Discriminator forward (force fp32 path internally)
            d_real = self.call_descriminator(tf.cast(y_target, tf.float32), training=True)
            d_fake = self.call_descriminator(tf.cast(y_pred,   tf.float32), training=True)

            # Discriminator / adversarial losses in fp32 with clipping
            d_loss  = self.d_loss_fn(d_real, d_fake)
            adv_loss = self.g_adv_loss_fn(d_fake)
            
            recon_loss = self.recon_loss_fn(y_target, y_pred)
            id_loss = (1.0) * self.id_loss_fn(y_target, y_pred) + (0.02) *self.id_loss_fn(y_target_up, y_pred_sr)
            #if self.upsampler:
                #recon_loss = upsample_loss + self.recon_loss_fn(y_target, y_pred)
                #id_loss = self.id_loss_fn(y_target_up, y_pred_sr) + self.id_loss_fn(y_target, y_pred)
                #sharp_loss = self.sharpness_loss_fn(y_target_up, y_pred) + self.sharpness_loss_fn(y_target, y_pred)
            #else:
                #recon_loss = self.recon_loss_fn(y_target, y_pred)
                #id_loss = self.id_loss_fn(y_target, y_pred)
                #sharp_loss = self.sharpness_loss_fn(y_target, y_pred)
            

            # Cast to float32 explicitly if needed
            adv_loss = tf.cast(adv_loss, tf.float32)
            recon_loss = tf.cast(recon_loss, tf.float32)
            id_loss = tf.cast(id_loss, tf.float32)
            upsample_loss = tf.cast(upsample_loss, tf.float32)
            color_loss = tf.cast(color_loss, tf.float32)

            # check each one for NaNs or Infs
            components = {
                'adv_loss':      adv_loss,
                'recon_loss':    recon_loss,
                'id_loss':       id_loss,
                'upsample_loss': upsample_loss,
                'color_loss':    color_loss
            }
            for name, tensor in components.items():
                tf.debugging.check_numerics(tensor, message=f"NaN or Inf in {name}")


            g_loss = (
                self.lambda_adv * adv_loss
                + self.lambda_recon * recon_loss
                + self.lambda_id * id_loss
                + self.lambda_upsample * upsample_loss
                + self.lambda_color * color_loss
                #+ self.lambda_sharp * sharp_loss
            )

        #tf.print("##### GEN VARNAMES:", [v.name for v in self.generator.trainable_variables])
        #tf.print("##### UPSAMPLER VARNAMES:", [v.name for v in self.upsampler.trainable_variables])


        # 4) Compute gradients for D and G
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)

        # 5) Apply gradients
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # Since we're done using the tape, delete if memory is a concern
        del tape
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.recon_loss_metric.update_state(recon_loss)
        self.adv_loss_metric.update_state(adv_loss)
        self.id_loss_metric.update_state(id_loss)
        self.up_loss_metric.update_state(upsample_loss)
        self.color_loss_metric.update_state(color_loss)

        #self.sharp_loss_metric.update_state(sharp_loss)


        # Return dictionary for display in logs
        return {
            "g_loss": self.g_loss_metric.result(),
            "recon_loss": self.recon_loss_metric.result(),
            "id_loss": self.id_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "adv_loss": self.adv_loss_metric.result(),
            "up_loss": self.up_loss_metric.result(),
            "color_loss": self.color_loss_metric.result(),
            #"sharp_loss": self.sharp_loss_metric.result(),
        } 
    
    def train_step256x256_baseline(self, data):
        # Check if data is a tuple or list
        if not isinstance(data, (tuple, list)):
            print(f"Expected data to be a tuple or list, got: {type(data)}")
            raise ValueError(f"Expected data to be a tuple or list, got: {type(data)}")

        # We can handle 2 types of training
        # 1) Complete prepreocessed of shape => (img_batch, embed_batch), (y_target, y_target_x2)
        # 2) No preprocessing  of shape => (img_batch, extract_embeds_from)
        # If type #2 then we will perform embedding extraction and use self.teacher to get y_target and y_target_x2
        if len(data) == 2:
            if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                (img_batch, embed_batch), (y_target, y_target_x2) = data
                extract_embeds_from = None
            else:
                img_batch, extract_embeds_from = data
                embed_batch = self.extract_normed_embed(extract_embeds_from)
                if self.teacher == None:
                    raise ValueError("Teacher model can't be none for this dataset shape!")
                y_target = self.teacher([img_batch, embed_batch], training=False)
                y_target_x2 = None            
        else:
            print(f"Unexpected data structure length: {len(data)}")
            raise ValueError(f"Unexpected data structure length: {len(data)}")


        with tf.GradientTape(persistent=True) as tape:
            # Forward generator
            embed_batch = self.extract_normed_embed(img_batch)
            y_pred_sr, y_pred, _ = self.call_generator([img_batch, embed_batch], training=True)

            y_pred_sr = tf.cast(y_pred_sr, tf.float16)
            y_pred = tf.cast(y_pred, tf.float16)
            y_target = tf.cast(img_batch, tf.float16)

            y_target_up = tf.image.resize(y_target, size=(512,512), method=tf.image.ResizeMethod.BILINEAR)
            #y_target_up = tf.image.resize(y_pred, size=(512,512), method=tf.image.ResizeMethod.BILINEAR) #I want to predict upsampling of the output only with this one
            y_target_up = tf.cast(y_target_up, tf.float16)

            color_loss = (0.55) * self.color_loss_fn(y_target, y_pred) + (0.45) * self.color_loss_fn(y_target_up, y_pred_sr, color_lambda=0.0, brightness_lambda=2.0)

            # Target upsampled data
            if self.upsampler:
                y_target_up = self.upsample_fn(y_target)
                y_target_up = tf.cast(y_target_up, tf.float16)
                
                y_target_up.set_shape((None, y_target.shape[1], y_target.shape[2], 3))  
                upsample_loss = self.recon_loss_fn(y_target_up, y_pred_sr)
            else:
                #upsample_loss = 0
                upsample_loss = self.recon_loss_fn(y_target_up, y_pred_sr)


            # Discriminator forward (force fp32 path internally)
            d_real = self.call_descriminator(tf.cast(y_target, tf.float32), training=True)
            d_fake = self.call_descriminator(tf.cast(y_pred,   tf.float32), training=True)

            # Discriminator / adversarial losses in fp32 with clipping
            d_loss  = self.d_loss_fn(d_real, d_fake)
            adv_loss = self.g_adv_loss_fn(d_fake)
            
            recon_loss = self.recon_loss_fn(y_target, y_pred)
            id_loss = (1.0) * self.id_loss_fn(y_target, y_pred) + (0.02) *self.id_loss_fn(y_target_up, y_pred_sr)

            # Cast to float32 explicitly if needed
            adv_loss = tf.cast(adv_loss, tf.float32)
            recon_loss = tf.cast(recon_loss, tf.float32)
            id_loss = tf.cast(id_loss, tf.float32)
            upsample_loss = tf.cast(upsample_loss, tf.float32)
            color_loss = tf.cast(color_loss, tf.float32)

            # check each one for NaNs or Infs
            components = {
                'adv_loss':      adv_loss,
                'recon_loss':    recon_loss,
                'id_loss':       id_loss,
                'upsample_loss': upsample_loss,
                'color_loss':    color_loss
            }
            for name, tensor in components.items():
                tf.debugging.check_numerics(tensor, message=f"NaN or Inf in {name}")


            g_loss = (
                self.lambda_adv * adv_loss
                + self.lambda_recon * recon_loss
                + self.lambda_id * id_loss
                + self.lambda_upsample * upsample_loss
                + self.lambda_color * color_loss
                #+ self.lambda_sharp * sharp_loss
            )

        #tf.print("##### GEN VARNAMES:", [v.name for v in self.generator.trainable_variables])
        #tf.print("##### UPSAMPLER VARNAMES:", [v.name for v in self.upsampler.trainable_variables])


        # 4) Compute gradients for D and G
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)

        # 5) Apply gradients
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # Since we're done using the tape, delete if memory is a concern
        del tape
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.recon_loss_metric.update_state(recon_loss)
        self.adv_loss_metric.update_state(adv_loss)
        self.id_loss_metric.update_state(id_loss)
        self.up_loss_metric.update_state(upsample_loss)
        self.color_loss_metric.update_state(color_loss)

        #self.sharp_loss_metric.update_state(sharp_loss)


        # Return dictionary for display in logs
        return {
            "g_loss": self.g_loss_metric.result(),
            "recon_loss": self.recon_loss_metric.result(),
            "id_loss": self.id_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "adv_loss": self.adv_loss_metric.result(),
            "up_loss": self.up_loss_metric.result(),
            "color_loss": self.color_loss_metric.result(),
            #"sharp_loss": self.sharp_loss_metric.result(),
        } 

    def train_baseline_OLD_REMOVE(self, data):
        # Check if data is a tuple or list
        if not isinstance(data, (tuple, list)):
            print(f"Expected data to be a tuple or list, got: {type(data)}")
            raise ValueError(f"Expected data to be a tuple or list, got: {type(data)}")

        # We can handle 2 types of training
        # 1) Complete prepreocessed of shape => (img_batch, embed_batch), (y_target, y_target_x2)
        # 2) No preprocessing  of shape => (img_batch, extract_embeds_from)
        # If type #2 then we will perform embedding extraction and use self.teacher to get y_target and y_target_x2
        if len(data) == 2:
            if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                (img_batch, embed_batch), (y_target, y_target_x2) = data
                extract_embeds_from = None
            else:
                img_batch, extract_embeds_from = data
                embed_batch = self.extract_normed_embed(extract_embeds_from)
                if self.teacher == None:
                    raise ValueError("Teacher model can't be none for this dataset shape!")
                y_target = self.teacher([img_batch, embed_batch], training=False)
                y_target_x2 = None            
        else:
            print(f"Unexpected data structure length: {len(data)}")
            raise ValueError(f"Unexpected data structure length: {len(data)}")


        with tf.GradientTape(persistent=True) as tape:
            # Forward generator
            embed_batch = self.extract_normed_embed(img_batch)
            y_pred_sr, y_pred, _ = self.call_generator([img_batch, embed_batch], training=True)

            y_pred = tf.cast(y_pred, tf.float16)
            y_target = tf.cast(img_batch, tf.float16) #When bootstrapping we want to match the input image

            if not self.generator.is_sr:
                id_loss = (1.0) * self.id_loss_fn(y_target, y_pred)
            else:
                y_pred_sr = tf.cast(y_pred_sr, tf.float16)
                y_target_up = tf.image.resize(y_target, size=(512,512), method=tf.image.ResizeMethod.BILINEAR)
                y_target_up = tf.cast(y_target_up, tf.float16)
                id_loss = (1.0) * self.id_loss_fn(y_target, y_pred) + (0.02) * self.id_loss_fn(y_target_up, y_pred_sr)
            
            d_real = self.call_descriminator(y_target, training=True)
            d_fake = self.call_descriminator(y_pred, training=True)

            # Discriminator loss
            d_loss = self.d_loss_fn(d_real, d_fake)

            # Generator losses
            adv_loss = self.g_adv_loss_fn(d_fake)

            recon_loss = self.recon_loss_fn(y_target, y_pred)
            
            # Cast to float32 explicitly if needed
            id_loss = tf.cast(id_loss, tf.float32)
            recon_loss = tf.cast(recon_loss, tf.float32)
            adv_loss = tf.cast(adv_loss, tf.float32)
            d_loss = tf.cast(d_loss, tf.float32)

            # check each one for NaNs or Infs
            components = {
                'id_loss':       id_loss,
                'recon_loss':    recon_loss,
                'adv_loss':      adv_loss,
                'd_loss':        d_loss,
            }
            for name, tensor in components.items():
                tf.debugging.check_numerics(tensor, message=f"NaN or Inf in {name}")


            g_loss = (
                self.lambda_id * id_loss,
                self.lambda_recon * recon_loss,
                self.lambda_adv * adv_loss,
            )

        #tf.print("##### GEN VARNAMES:", [v.name for v in self.generator.trainable_variables])
        #tf.print("##### UPSAMPLER VARNAMES:", [v.name for v in self.upsampler.trainable_variables])


        # 4) Compute gradients for D and G
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)

        # 5) Apply gradients
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # Since we're done using the tape, delete if memory is a concern
        del tape
        
        # Update metrics
        
        self.g_loss_metric.update_state(g_loss)
        self.id_loss_metric.update_state(id_loss)
        self.recon_loss_metric.update_state(recon_loss)
        self.d_loss_metric.update_state(d_loss)
        self.adv_loss_metric.update_state(adv_loss)
        #self.sharp_loss_metric.update_state(sharp_loss)


        # Return dictionary for display in logs
        return {
            "g_loss": self.g_loss_metric.result(),
            "recon_loss": self.recon_loss_metric.result(),
            "id_loss": self.id_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "adv_loss": self.adv_loss_metric.result(),
            "up_loss": tf.constant(0.0, dtype=tf.float32),
            "color_loss": tf.constant(0.0, dtype=tf.float32),
            #"sharp_loss": self.sharp_loss_metric.result(),
        } 
    
    def test_step(self, data):
        """
        If you provide validation_data=... to model.fit, 
        Keras will call test_step on each batch of val data.
        We'll measure just reconstruction loss as 'val_loss' for example.
        """
        (img_batch, embed_batch),  (y_target, y_target_x2) = data

        # forward pass generator in inference mode
        y_pred, _ = self.generator([img_batch, embed_batch], training=False)
        # compute identity loss
        id_loss = self.id_loss_fn(y_target, y_pred)

        return {"val_loss": id_loss}
    
    def call(self, inputs, training=None, mask=None):
        """
        This method is used when you do `aei_gan_model([imgs, embeds])`.
        We'll just forward to the generator and return its output.
        """
        # inputs should be `[img_batch, embed_batch]` as in your code.
        # AEI-Net typically returns `(Y_pred, z_attr)`.
        y_pred, _, z_attr = self.generator(inputs, training=training)
        # Return the generated images for inference
        return y_pred
    
    def save(self, *args, **kwargs):
        temp_teacher = None
        # Temporarily remove teacher before saving
        if self.teacher != None:
            try:
                temp_teacher = self.teacher
                self.teacher = None
            except Exception as e:
                print(e)
                pass
        
        result = super().save(*args, **kwargs)
        
        # Restore teacher after saving
        if temp_teacher:
            self.teacher = temp_teacher

        return result

