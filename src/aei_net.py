
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, BatchNormalization,
                                     LeakyReLU, Add, Concatenate, Dense, Activation,
                                     UpSampling2D, ReLU, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotNormal
import tensorflow_addons as tfa
from sr_model import FaceSuperResolutionModel
from simple_weighted_skip import simple_weighted_skip

# Weight initializer
def get_weight_initializer():
    return GlorotNormal()

# Convolutional block
def conv4x4(out_channels, norm_layer=BatchNormalization):
    def layer(x):
        x = Conv2D(filters=out_channels, kernel_size=4, strides=2, padding='same', use_bias=False,
                   kernel_initializer=get_weight_initializer())(x)
        x = norm_layer()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x
    return layer

class SubPixelUpsampler(tf.keras.layers.Layer):
    """
    Learns how to upsample via a sub-pixel convolution:
    1) Apply a Conv2D to increase channels by (scale^2)
    2) Reshape channels into a higher spatial resolution
       using depth_to_space.
    """
    def __init__(self, out_channels, scale=2, kernel_size=3, 
                 activation=None, name=None, **kwargs):
        if name:
            super(SubPixelUpsampler, self).__init__(name=name, **kwargs)
        else:
            super(SubPixelUpsampler, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.scale = scale
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = Conv2D(
            filters=self.out_channels * (self.scale ** 2),
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=GlorotNormal(),
            activation=self.activation,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "scale": self.scale,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "conv": self.conv
        })
        return config

    def call(self, x):
        """
        x shape: (batch, H, W, C)
        After conv: (batch, H, W, out_channels * scale^2)
        After depth_to_space: (batch, H*scale, W*scale, out_channels)
        """
        x = self.conv(x)  # shape => (B, H, W, out_channels*(scale^2))
        x = tf.nn.depth_to_space(x, block_size=self.scale)
        return x

    # Deconvolutional block
class Deconv4x4(tf.keras.layers.Layer):
    """
    Instead of a transposed convolution, we do:
      x = UpSampling2D(...) -> Conv2D(3x3) -> BN -> LeakyReLU
    """
    def __init__(self, 
                 out_channels, 
                 skip_channels,
                 norm_layer=BatchNormalization, 
                 backbone='unet', 
                 scale=2, 
                 **kwargs):
        super(Deconv4x4, self).__init__(**kwargs)
        self.backbone = backbone
        self.scale = scale
        
        # Replace Conv2DTranspose with UpSampling2D + Conv2D
        self.upsampler = SubPixelUpsampler(
            out_channels=out_channels,  # final channels after upsampling
            scale=self.scale,
            kernel_size=3,
            activation=None
        )
        
        self.bn = norm_layer()
        self.lrelu = LeakyReLU(alpha=0.1)
        
        # 1x1 convolution to adjust channels after concatenation
        self.conv1x1 = Conv2D(
            filters=out_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=get_weight_initializer()
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": self.backbone,
            "scale": self.scale,
            "upsampler": self.upsampler,
            "bn": self.bn,
            "lrelu": self.lrelu,
            "conv1x1": self.conv1x1,
        })
        return config

    def call(self, inputs):
        x_input, skip_input = inputs
        
        # Up-sample, then convolve
        x = self.upsampler(x_input)
        x = self.bn(x)
        x = self.lrelu(x)
        
        # Merge skip-connection
        if self.backbone == 'linknet':
            x = Add()([x, skip_input])
        else:
            x = Concatenate(axis=-1)([x, skip_input])
            x = self.conv1x1(x)  # Adjust channels back to out_channels

        #More skip more better?
        #x = simple_weighted_skip([x, skip_input], skip_alpha=0.45)

        return x

    
# AADLayer implementation
class AADLayer(tf.keras.layers.Layer):
    def __init__(self, c_x, attr_c, c_id, name=None, **kwargs):
        if name:
            super(AADLayer, self).__init__(name=name, **kwargs)
        else:
            super(AADLayer, self).__init__(**kwargs)
        self.c_x = c_x
        self.conv1 = Conv2D(filters=c_x, kernel_size=1, strides=1, padding='valid', use_bias=True, name=name)
        self.conv2 = Conv2D(filters=c_x, kernel_size=1, strides=1, padding='valid', use_bias=True, name=name)
        self.fc1 = Dense(c_x, name=name)
        self.fc2 = Dense(c_x, name=name)
        self.norm = tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False, name=name)
        self.conv_h = Conv2D(filters=1, kernel_size=1, strides=1, padding='valid', use_bias=True, name=name)

    def get_config(self):
        config = super().get_config()
        config.update({
            "c_x": self.c_x,
            "conv1": self.conv1,
            "conv2": self.conv2,
            "fc1": self.fc1,
            "fc2": self.fc2,
            "norm": self.norm,
            "conv_h": self.conv_h,
        })
        return config

    def call(self, inputs):
        h_in, z_attr, embed = inputs
        h = self.norm(h_in)
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)
        gamma_id = self.fc1(embed)
        beta_id = self.fc2(embed)
        gamma_id = tf.reshape(gamma_id, [-1, 1, 1, self.c_x])
        beta_id = tf.reshape(beta_id, [-1, 1, 1, self.c_x])
        gamma_id = tf.tile(gamma_id, [1, tf.shape(h)[1], tf.shape(h)[2], 1])
        beta_id = tf.tile(beta_id, [1, tf.shape(h)[1], tf.shape(h)[2], 1])
        A = gamma_attr * h + beta_attr
        I = gamma_id * h + beta_id
        M = Activation('sigmoid')(self.conv_h(h))
        out = (1 - M) * A + M * I
        return out

# AAD ResBlock
def AAD_ResBlk(cin, cout, c_attr, c_id, num_blocks, name=None):
    def layer(inputs):
        h, z_attr, embed = inputs
        x = h
        for i in range(num_blocks):
            name_i = f"{name}_x_{i}" if name else None
            out_channels = cin if i < (num_blocks - 1) else cout
            x = AADLayer(cin, c_attr, c_id, name=name_i)([x, z_attr, embed])
            x = ReLU()(x)
            x = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same', use_bias=False,
                       kernel_initializer=get_weight_initializer())(x)
        if cin != cout:
            name_i = f"{name}_h_{i}" if name else None
            h = AADLayer(cin, c_attr, c_id, name=name_i)([h, z_attr, embed])
            h = ReLU()(h)
            h = Conv2D(filters=cout, kernel_size=3, strides=1, padding='same', use_bias=False,
                       kernel_initializer=get_weight_initializer())(h)
        
        x = Add()([x, h])

        return x
    return layer

# Adjusted MLAttrEncoder for input shape (64, 64, 3)
def MLAttrEncoder64(input_shape, backbone='unet'):
    Xt = Input(shape=input_shape)
    feat1 = conv4x4(32)(Xt)        # Output: (batch_size, 32, 32, 32)
    feat2 = conv4x4(64)(feat1)     # Output: (batch_size, 16, 16, 64)
    feat3 = conv4x4(128)(feat2)    # Output: (batch_size, 8, 8, 128)
    feat4 = conv4x4(256)(feat3)    # Output: (batch_size, 4, 4, 256)
    z_attr1 = conv4x4(512)(feat4)  # Output: (batch_size, 2, 2, 512)

    # Deconvolutional layers with 1x1 convolutions to adjust channels
    z_attr2 = Deconv4x4(512, 256, backbone=backbone)([z_attr1, feat4])
    z_attr3 = Deconv4x4(256, 128, backbone=backbone)([z_attr2, feat3])
    z_attr4 = Deconv4x4(128, 64, backbone=backbone)([z_attr3, feat2])
    z_attr5 = Deconv4x4(64, 32, backbone=backbone)([z_attr4, feat1])
    z_attr6 = UpSampling2D(size=2, interpolation='bilinear')(z_attr5)
    encoder_model = Model(
        inputs=Xt,
        outputs=[z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6]
    )
    return encoder_model

def MLAttrEncoder256(input_shape=(256, 256, 3), backbone='unet'):
    Xt = Input(shape=input_shape)

    # Downsampling Path (Feature Extraction)
    feat1 = conv4x4(32)(Xt)        # Output: (batch_size, 128, 128, 32)
    feat2 = conv4x4(64)(feat1)     # Output: (batch_size, 64, 64, 64)
    feat3 = conv4x4(128)(feat2)    # Output: (batch_size, 32, 32, 128)
    feat4 = conv4x4(256)(feat3)    # Output: (batch_size, 16, 16, 256)
    feat5 = conv4x4(512)(feat4)    # Output: (batch_size, 8, 8, 512)
    feat6 = conv4x4(512)(feat5)    # Output: (batch_size, 4, 4, 512)
    z_attr1 = conv4x4(512)(feat6)  # Output: (batch_size, 2, 2, 512)

    # Upsampling Path (Deconvolution)
    z_attr2 = Deconv4x4(512, 512, backbone=backbone)([z_attr1, feat6])
    z_attr3 = Deconv4x4(512, 256, backbone=backbone)([z_attr2, feat5])
    z_attr4 = Deconv4x4(256, 128, backbone=backbone)([z_attr3, feat4])
    z_attr5 = Deconv4x4(128, 64, backbone=backbone)([z_attr4, feat3])
    z_attr6 = Deconv4x4(64, 32, backbone=backbone)([z_attr5, feat2])
    z_attr7 = Deconv4x4(32, 16, backbone=backbone)([z_attr6, feat1])
    z_attr8 = UpSampling2D(size=2, interpolation='bilinear')(z_attr7)  # Output: (batch_size, 256, 256, 16)
    #z_attr8 = SubPixelUpsampler(out_channels=32, scale=2, kernel_size=3, activation=None, name="z_attr8_upsampler")(z_attr7)  # Output: (batch_size, 256, 256, 16)
    


    encoder_model = Model(
        inputs=Xt,
        outputs=[z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8]
    )

    return encoder_model

# Adjusted AADGenerator for fewer layers
def AADGenerator64(c_id=512, num_blocks=2, backbone='unet'):
    z_attr_channels = [512, 512, 256, 128, 64, 64]
    z_attr_inputs = [Input(shape=(None, None, c)) for c in z_attr_channels]
    embed_input = Input(shape=(c_id,))
    
    embed_reshaped = tf.reshape(embed_input, [-1, 1, 1, c_id])
    m1 = embed_reshaped
    #m1 = Conv2DTranspose(
    #    filters=512,
    #    kernel_size=2,
    #    strides=1,
    #    padding='valid',
    #    kernel_initializer=get_weight_initializer()
    #)(embed_reshaped)
    y1 = AAD_ResBlk(512, 512, z_attr_channels[0], c_id, num_blocks)([m1, z_attr_inputs[0], embed_input])
    m2 = UpSampling2D(size=2, interpolation='bilinear')(y1)
    y2 = AAD_ResBlk(512, 256, z_attr_channels[1], c_id, num_blocks)([m2, z_attr_inputs[1], embed_input])
    m3 = UpSampling2D(size=2, interpolation='bilinear')(y2)
    y3 = AAD_ResBlk(256, 128, z_attr_channels[2], c_id, num_blocks)([m3, z_attr_inputs[2], embed_input])
    m4 = UpSampling2D(size=2, interpolation='bilinear')(y3)
    y4 = AAD_ResBlk(128, 64, z_attr_channels[3], c_id, num_blocks)([m4, z_attr_inputs[3], embed_input])
    m5 = UpSampling2D(size=2, interpolation='bilinear')(y4)
    y5 = AAD_ResBlk(64, 64, z_attr_channels[4], c_id, num_blocks)([m5, z_attr_inputs[4], embed_input])
    m6 = UpSampling2D(size=2, interpolation='bilinear')(y5)
    y6 = AAD_ResBlk(64, 3, z_attr_channels[5], c_id, num_blocks)([m6, z_attr_inputs[5], embed_input])
    y6 = Activation('tanh', dtype='float32')(y6)
    generator_model = Model(inputs=[z_attr_inputs, embed_input], outputs=y6)
    return generator_model

# Adjusted AADGenerator to output 256x256 images
def AADGenerator256(c_id=512, num_blocks=2, backbone='unet'):    
    z_attr_channels = [512, 512, 512, 256, 128, 64, 32, 32]

    # Define inputs for each resolution level
    z_attr_inputs = [Input(shape=(None, None, c)) for c in z_attr_channels]
    embed_input = Input(shape=(c_id,))
    
    # Reshape the embedding input
    embed_reshaped = tf.reshape(embed_input, [-1, 1, 1, c_id])
    
    # First Residual Block at 4x4
    y1 = AAD_ResBlk(512, 512, z_attr_channels[0], c_id, num_blocks)([embed_reshaped, z_attr_inputs[0], embed_input])

    # Upsample to 8x8
    m2 = UpSampling2D(size=2, interpolation='bilinear')(y1)
    y2 = AAD_ResBlk(512, 256, z_attr_channels[1], c_id, num_blocks)([m2, z_attr_inputs[1], embed_input])
    
    # Upsample to 16x16
    m3 = UpSampling2D(size=2, interpolation='bilinear')(y2)
    y3 = AAD_ResBlk(256, 128, z_attr_channels[2], c_id, num_blocks)([m3, z_attr_inputs[2], embed_input])
    
    # Upsample to 32x32
    m4 = UpSampling2D(size=2, interpolation='bilinear')(y3)
    y4 = AAD_ResBlk(128, 64, z_attr_channels[3], c_id, num_blocks)([m4, z_attr_inputs[3], embed_input])
    
    # Upsample to 64x64
    m5 = UpSampling2D(size=2, interpolation='bilinear', name="upsample_m5_unfreeze")(y4)
    y5 = AAD_ResBlk(64, 64, z_attr_channels[4], c_id, num_blocks, name="addresblk_y5_unfreeze")([m5, z_attr_inputs[4], embed_input])
    
    # Upsample to 128x128
    m6 = UpSampling2D(size=2, interpolation='bilinear', name="upsample_m6_unfreeze")(y5)
    y6 = AAD_ResBlk(64, 32, z_attr_channels[5], c_id, num_blocks, name="addresblk_y6_unfreeze")([m6, z_attr_inputs[5], embed_input])
    
    # Upsample to 256x256 (NEW)
    m7 = UpSampling2D(size=2, interpolation='bilinear', name="upsample_m7_unfreeze")(y6)
    y7 = AAD_ResBlk(32, 16, z_attr_channels[6], c_id, num_blocks, name="addresblk_y7_unfreeze")([m7, z_attr_inputs[6], embed_input])
    
    # Final upsampling to 256x256
    m8 = UpSampling2D(size=2, interpolation='bilinear', name="upsample_m8_unfreeze")(y7)
    y8 = AAD_ResBlk(16, 3, z_attr_channels[7], c_id, num_blocks, name="addresblk_y8_unfreeze")([m8, z_attr_inputs[7], embed_input])

    # Final activation to scale to [-1,1] range
    y8 = Activation('tanh', dtype='float32')(y8)

    # Create the model
    generator_model = Model(inputs=[z_attr_inputs, embed_input], outputs=y8)
    
    return generator_model

def freeze_except_unfreeze_layers(model, freeze_only_superres=False, keyword="unfreeze"):
    # Recursively freeze all layers.
    def set_trainable_recursive(layer, trainable):
        layer.trainable = trainable
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                set_trainable_recursive(sub_layer, trainable)
    
    # Freeze everything first.
    set_trainable_recursive(model, False)
    
    if freeze_only_superres:
        sr_submodel = model.get_layer('sr_model')
        sr_submodel.trainable = True
        for layer in sr_submodel.layers:
            layer.trainable = True
    else:
        # Unfreeze layers whose names contain the keyword.
        for layer in model.submodules:
            if keyword in layer.name:
                layer.trainable = True

def freeze_all_except_deconv4x4(model):
    # Recursive helper to freeze all layers
    def set_trainable_recursive(layer, trainable):
        layer.trainable = trainable
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                set_trainable_recursive(sub_layer, trainable)
    
    # Freeze every layer in the model.
    set_trainable_recursive(model, False)
    
    # Unfreeze only layers of type Deconv4x4.
    for layer in model.submodules:
        if isinstance(layer, Deconv4x4):
            layer.trainable = True

def radial_skip(tensors, edge_power=1.5):
    Y, X = tensors
    shape = tf.shape(Y)
    H = tf.cast(shape[1], tf.float32)
    W = tf.cast(shape[2], tf.float32)

    # coords in [-1,1]
    ys = tf.linspace(-1.0, 1.0, shape[1])
    xs = tf.linspace(-1.0, 1.0, shape[2])
    grid_x, grid_y = tf.meshgrid(xs, ys)         # both [H,W]

    # radial distance normalized to [0,1]
    R = tf.sqrt(grid_x**2 + grid_y**2)
    R = tf.clip_by_value(R / tf.sqrt(2.0), 0.0, 1.0)

    # edge sharpening
    alpha_map = R ** edge_power                  # [H,W]

    # broadcast to [B,H,W,1]
    alpha_map = tf.expand_dims(alpha_map, axis=0)
    alpha_map = tf.expand_dims(alpha_map, axis=-1)

    alpha_map = tf.cast(alpha_map, Y.dtype)

    # blend: center favors Y, edges favor X
    return Y * (1.0 - alpha_map) + X * alpha_map

def match_mean_brightness(src_rgb, tgt_rgb, eps=1e-6):
    """
    Scale `tgt_rgb` so its mean luminance (Y channel) equals that of `src_rgb`.
    Works on a single image or a batch in [0,1] float32/16.
    """
    src_y = tf.image.rgb_to_yuv(src_rgb)[..., :1]          # (N,H,W,1)
    tgt_y = tf.image.rgb_to_yuv(tgt_rgb)[..., :1]

    src_mean = tf.reduce_mean(src_y, axis=[1,2,3], keepdims=True)
    tgt_mean = tf.reduce_mean(tgt_y, axis=[1,2,3], keepdims=True)

    src_mean = tf.cast(src_mean, tf.float16)
    tgt_mean = tf.cast(tgt_mean, tf.float16)

    scale = src_mean / (tgt_mean + eps)                   # (N,1,1,1)
    out = tf.clip_by_value(tgt_rgb * scale, 0.0, 1.0)
    return out

# Adjusted AEI_Net with Super Resolution
def AEI_Net256SR(input_shape=(256, 256, 3), c_id=512, num_blocks=2, backbone='unet'):
    encoder_model = MLAttrEncoder256(input_shape, backbone)
    encoder_model.summary()
    generator_model = AADGenerator256(c_id, num_blocks, backbone)
    generator_model.summary()
    Xt_input = encoder_model.input
    z_attr = encoder_model.output
    embed_input = Input(shape=(c_id,))
    Y = generator_model([z_attr, embed_input])
    Y_with_skip = simple_weighted_skip([Y, encoder_model.input], 0.15, "1")
    #skip_alpha=0.45
    #Y_with_skip = Lambda(
    #    lambda tensors: tensors[0] * (1.0 - skip_alpha)
    #                    + tensors[1] * skip_alpha,
    #    name="scaled_skip"
    #)([Y, encoder_model.input])
    #Y_with_skip = Lambda(
    #    lambda t: tf.clip_by_value(t, 0.0, 1.0),
    #    name="clamp"
    #)(Y_with_skip)
    #Y_with_skip = Lambda(radial_skip, name="radial_skip")([Y, Xt_input])
    #Y_with_skip = Lambda(lambda t: tf.clip_by_value(t, 0.0, 1.0), name="clamp")(Y_with_skip)
    sr_model = FaceSuperResolutionModel(name="sr_model")
    sr = sr_model(Y_with_skip)
    
    sr_bc = Lambda(
        lambda args: match_mean_brightness(args[0], args[1]),
        name="brightness_match_sr"
    )([Xt_input, sr])

    Y_bc = Lambda(
        lambda args: match_mean_brightness(args[0], args[1]),
        name="brightness_match_y"
    )([Xt_input, Y])

    model = Model(inputs=[Xt_input, embed_input], outputs=[sr_bc, Y_bc, z_attr])
    model.is_sr = True
    return model

# Adjusted AEI_Net
def AEI_Net256(input_shape=(256, 256, 3), c_id=512, num_blocks=2, backbone='unet'):
    encoder_model = MLAttrEncoder256(input_shape, backbone)
    encoder_model.summary()
    generator_model = AADGenerator256(c_id, num_blocks, backbone)
    generator_model.summary()
    Xt_input = encoder_model.input
    z_attr = encoder_model.output
    embed_input = Input(shape=(c_id,))
    Y = generator_model([z_attr, embed_input])
    model = Model(inputs=[Xt_input, embed_input], outputs=[Y, Y, z_attr])
    model.is_sr = False
    return model

def AEI_Net64(input_shape=(64, 64, 3), c_id=512, num_blocks=2, backbone='unet'):
    encoder_model = MLAttrEncoder64(input_shape, backbone)
    encoder_model.summary()
    generator_model = AADGenerator64(c_id, num_blocks, backbone)
    generator_model.summary()
    Xt_input = encoder_model.input
    z_attr = encoder_model.output
    embed_input = Input(shape=(c_id,))
    Y = generator_model([z_attr, embed_input])
    model = Model(inputs=[Xt_input, embed_input], outputs=[Y, z_attr])
    return model

def get_model(input_shape=(256, 256, 3), c_id=512, num_blocks=2, freeze=False, freeze_all_except_deconv=False, with_super_resolution=False):
    if input_shape[0] == 64:
        model = AEI_Net64(input_shape, c_id=512, num_blocks=num_blocks)    
    elif with_super_resolution == True:
        model = AEI_Net256SR(input_shape, c_id=512, num_blocks=num_blocks)
    else:
        model = AEI_Net256(input_shape, c_id=512, num_blocks=num_blocks)


    if freeze:
        freeze_except_unfreeze_layers(model, freeze_only_superres=with_super_resolution)
    
    if freeze_all_except_deconv:
        freeze_all_except_deconv4x4(model)

    return model

