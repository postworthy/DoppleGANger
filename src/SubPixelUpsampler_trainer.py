import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from aei_net import SubPixelUpsampler
from data_loader import get_training_data
from AEIGANModel import AEIGANModel
from aei_net import get_model
from discriminator import MinimalPatchGAN
from tqdm import tqdm
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
NUM_FEATURES = 512
Z_DIM = 200
LEARNING_RATE = 1e-4 #0.0005
WEIGHT_DECAY = 1e-7
EPOCHS = 5
BETA = 2000
LOAD_MODEL = True
TAKE_BATCHES = 500
VALIDATION_SET_SIZE = 10
NUM_BLOCKS=2

model = AEIGANModel(
    generator=get_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_blocks=NUM_BLOCKS),
    discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
    lambda_recon=150.0, lambda_adv=0.5, lambda_id=45.0, lambda_upsample=100.0
)
model.trainable = False
# Load old weights if required
model.load_weights("./models/MODEL_256x256_v6_BLOCKS2_latest")

try:
    if model.generator.get_layer("z_attr8_upsampler"):
        print("###############################################################")
        print("You are trying to train with the z_attr8_upsampler version bro!")
        print("###############################################################")
        exit()
except:
    pass
finally:
    pass

base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

class CustomUpsamplerModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. It can be a tuple: ((img_batch, embed_batch), (y_target, y_target_x2))
        (img_batch, embed_batch), (y_target, y_target_x2) = data
        
        with tf.GradientTape() as tape:    
            # Forward pass through the generator (frozen)
            img, z_attr = model.generator([img_batch, embed_batch])
            z_attr7 = z_attr[6]
            z_attr8 = z_attr[7]

            # Forward pass through our upsampler (trainable)
            pred_z_attr8 = self(z_attr7, training=True)
            
            # Compute the loss (MSE between the target and prediction)
            loss = self.compiled_loss(
                tf.cast(z_attr8, tf.float32),
                tf.cast(pred_z_attr8, tf.float32),
                regularization_losses=self.losses
            )
            # Scale the loss for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        # Compute gradients using the scaled loss
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        # Unscale the gradients
        unscaled_gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(unscaled_gradients, self.trainable_variables))

        # Optionally update any metrics if needed.
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss  # report the unscaled loss
        return metrics



input_shape = (None, None, 32)
inputs = Input(shape=input_shape)
upsampler_layer = SubPixelUpsampler(out_channels=32, scale=2, kernel_size=3, activation=None, name="z_attr8_upsampler")
outputs = upsampler_layer(inputs)
upsampler_model = CustomUpsamplerModel(inputs=inputs, outputs=outputs)
upsampler_model.compile(optimizer=optimizer, loss='mse')
upsampler_model.load_weights("./models/upsampler_weights.h5")
for i in tqdm(range(500), desc="Outer Loop Progress"):
    train, validation = get_training_data(tfrecord_shard_path="./data/kitchensink_256/", p=0.01)
    dataset = validation.concatenate(train)
    upsampler_model.fit(dataset, epochs=5)

    if i % 5 == 0:
        upsampler_model.save_weights("./models/upsampler_weights.h5")

upsampler_model.save_weights("./models/upsampler_weights.h5")
