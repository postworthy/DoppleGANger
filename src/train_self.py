from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.65)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import mixed_precision
from data_loader import get_self_training_data_with_augmentation, get_training_data
from aei_net import get_model
from discriminator import MinimalPatchGAN
from training_callbacks import get_callbacks
from AEIGANModel import AEIGANModel
from super_resolution import SuperResModel
from tensorflow_addons.optimizers import AdamW
import math
import gc
import argparse
from tqdm import tqdm
from adaptive_loss import AdaptiveIdLossImprovement
import onnxruntime as ort
from onnx_upsampler import onnx_upsample
from ISR.models import RDN, RRDN

mixed_precision.set_global_policy('mixed_float16') #This was produucing NAN when training beyond ~7 epochs
#tf.config.optimizer.set_jit(True)

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
SAVE_EVERY_N_EPOCHS = 2
NUM_BLOCKS=2

#generator = get_model(num_blocks=NUM_BLOCKS)
#discriminator = MinimalPatchGAN(ndf=64, num_layers=3)

terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

class FinalModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_path, total_epochs):
        """
        Initialize the callback.
        :param save_path: Base path where the model will be saved.
        :param total_epochs: Total number of epochs for training, used to determine the last epoch.
        """
        super().__init__()
        self.save_path = save_path
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model only on:
        - First epoch (epoch == 0)
        - Last epoch (epoch == self.total_epochs - 1)
        - Every 5th epoch (epoch % 5 == 4)
        """
        if epoch == 0 or epoch == self.total_epochs - 1 or (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_path_with_epoch = f"{self.save_path}_epoch_{epoch + 1}"
            self.model.save(save_path_with_epoch)
            print(f"Model saved at: {save_path_with_epoch}")

        # Always save a "latest" checkpoint
        self.model.save(f"{self.save_path}_latest")
        print(f"Model saved at: {self.save_path}_latest")

def train(total_epochs=1, 
          reload_interval=1, 
          model_save_path="./models/aei_net_gan_weights/",
          model_weights="./models/aei_net_gan_kitchensink_128_v3_150_0.5_45_15_latest",
          tfrecord_shard_path="./data/kitchensink_128/"):
    
    print(f"###### Training for {total_epochs} epochs with reload interval {reload_interval}")
    
    #upsampler = RRDN(arch_params={'C':4, 'D':3, 'G':32, 'G0':32, 'T':10, 'x':4})
    #upsampler.model.load_weights("./isr_weights/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5")

    # Load aei-net model generator and discriminator
    teacher = AEIGANModel(
        generator=get_model(num_blocks=NUM_BLOCKS),
        discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
        lambda_recon=150.0, lambda_adv=0.5, lambda_id=45.0)
    teacher.compile(
        g_optimizer=AdamW(learning_rate=4e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
        d_optimizer=AdamW(learning_rate=4e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
    )

    # Load old weights if required
    if model_weights and os.path.isdir(model_weights):
        print(f"###### Using model weights: {model_weights}")
        teacher.load_weights(model_weights)
        teacher.g_optimizer=AdamW(learning_rate=1e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4) 
        teacher.d_optimizer=AdamW(learning_rate=1e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4)

    # Load aei-net model generator and discriminator
    model = AEIGANModel(
        generator=get_model(num_blocks=NUM_BLOCKS),
        discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
        teacher=teacher,
        #upsampler=upsampler.model,
        upsampler=onnx_upsample,
        lambda_recon=150.0, lambda_adv=10.0, lambda_id=45.0, lambda_upsample=20.0)
    model.compile(
        g_optimizer=AdamW(learning_rate=4e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
        d_optimizer=AdamW(learning_rate=4e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
    )

    # Load old weights if required
    if model_weights and os.path.isdir(model_weights):
        print(f"###### Using model weights: {model_weights}")
        model.load_weights(model_weights)
        model.g_optimizer=AdamW(learning_rate=1e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4) 
        model.d_optimizer=AdamW(learning_rate=1e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4)
    
    # Prime the models with fake data    
    dummy_img = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], dtype=tf.float16)
    dummy_embed = tf.zeros([1, NUM_FEATURES], dtype=tf.float16)
    model((dummy_img, dummy_embed), training=False)
    teacher((dummy_img, dummy_embed), training=False)
    
    # Calculate epoch / training dataset breakdown
    num_outer = math.ceil(total_epochs / reload_interval)
    epochs_per_fit = total_epochs // num_outer
    remaining_epochs = total_epochs - (epochs_per_fit * num_outer)
    
    #Adaptive Loss
    adaptive_id_loss_cb = AdaptiveIdLossImprovement(improvement_threshold=0.0001, patience=2, increase_factor=1.2)

    # Main training loop
    for i in tqdm(range(num_outer), desc="Outer Loop Progress"):
        # Cache clearing so we can get a full data set
        try:
            del train
            del validation
            gc.collect()
            tf.keras.backend.clear_session()
            print("###### Cleared previous training data")
        except NameError:
            pass
        
        # Get Training and validation data
        #train = get_self_training_data_with_augmentation(tfrecord_shard_path=tfrecord_shard_path, p=0.02)
        train, validation = get_training_data(batch_size=32, tfrecord_shard_path=tfrecord_shard_path, p=0.15)

        _, _, image_generator = get_callbacks(validation)
        # Epoch calculation incase we have uneven divisibility
        current_epochs = epochs_per_fit + (1 if i < remaining_epochs else 0)
        
        # aei-net training
        model.fit(
            train,
            epochs=current_epochs,
            callbacks=[
                FinalModelSaver(f"{model_save_path}_BLOCKS{NUM_BLOCKS}", current_epochs),
                terminate_on_nan,
                # model_checkpoint_callback,
                # tensorboard_callback,
                image_generator,
                #adaptive_id_loss_cb,
            ]
        )
        
        

    # Final Model Save
    model.save(f"{model_save_path}_BLOCKS{NUM_BLOCKS}_latest")
    print(f"Model saved at: {model_save_path}_BLOCKS{NUM_BLOCKS}_latest")
    

        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with specified parameters.")
    
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=10,
        help="Total number of epochs for training (default: 10)"
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=1,
        help="Interval for reloading the dataset (default: 1)"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="./models/aei_net_gan_adaptive_loss_weights_BLOCKS4_latest",
        help="Path to the model weights (default: './models/aei_net_gan_adaptive_loss_weights_BLOCKS4_latest')"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models/aei_net_gan_weights",
        help="Path to the super resolution model weights (default: './models/aei_net_gan_weights')"
    )
    parser.add_argument(
        "--tfrecord_shard_path",
        type=str,
        default="./data/kitchensink_128/",
        help="Path to the dataset (default: './data/kitchensink_128/')"
    )
    
    args = parser.parse_args()

    #tf.config.optimizer.set_jit(True)

    train(
        total_epochs=args.total_epochs,
        reload_interval=args.reload_interval,
        model_save_path=args.model_save_path,
        model_weights=args.model_weights,
        tfrecord_shard_path=args.tfrecord_shard_path
    )
