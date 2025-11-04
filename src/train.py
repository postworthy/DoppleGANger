from gpu_memory import limit_gpu_memory
limit_gpu_memory(0.85)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import mixed_precision
from data_loader import get_training_data
from aei_net import get_model, SubPixelUpsampler
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
from ISR.models import RDN, RRDN
from onnx_upsampler import Upsampler 

mixed_precision.set_global_policy('mixed_float16')
#tf.config.optimizer.set_jit(True)

IMAGE_SIZE = 256
CHANNELS = 3
DATASET_BATCH_SIZE = 32
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

#generator = get_model(num_blocks=NUM_BLOCKS)
#discriminator = MinimalPatchGAN(ndf=64, num_layers=3)

terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

class FinalModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_path, total_epochs, save_epoch):
        super().__init__()
        self.save_path = save_path
        self.total_epochs = total_epochs
        self.save_epoch=save_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_epoch == 0:
            save_path_with_epoch = f"{self.save_path}_epoch_{epoch + 1}"
            self.model.save(save_path_with_epoch)
            print(f"Model saved at: {save_path_with_epoch}")

            self.model.save(f"{self.save_path}_latest")
            print(f"Model saved at: {self.save_path}_latest")


def train(total_epochs=1, 
          reload_interval=1, 
          model_save_path="./models/aei_net_gan_weights/",
          model_weights="./models/aei_net_gan_kitchensink_128_v3_150_0.5_45_15_latest",
          train_superres=False,
          tfrecord_shard_path="./data/kitchensink_128/",
          save_epoch=1,
          new_discriminator=False, 
          deconv_only=False,
          randomize_shards=False,
          g_learning_rate=1e-4,
          d_learning_rate=1e-6,
          multi_gpu=False,
          finetune_superres=False,
          use_emap=False,
          num_blocks=NUM_BLOCKS,
          max_batches=None):
    
    print(f"###### Training for {total_epochs} epochs with reload interval {reload_interval}")
    
    #if train_superres:
        #upsampler=Upsampler().sr_model
        #upsampler = RRDN(arch_params={'C':4, 'D':3, 'G':32, 'G0':32, 'T':10, 'x':4})
        #upsampler.model.load_weights("./isr_weights/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5")
        #upsampler = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
        #upsampler.model.load_weights("./isr_weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5")
        #upsampler = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
        #upsampler.model.load_weights("./isr_weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5")
    #else:
        #upsampler=None

    upsampler=None

    # Load aei-net model generator and discriminator
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print("Multi-GPU enabled. Number of devices:", strategy.num_replicas_in_sync)
        with strategy.scope():           
            model = AEIGANModel(
                generator=get_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_blocks=num_blocks, freeze=finetune_superres, freeze_all_except_deconv=deconv_only, with_super_resolution=train_superres),
                discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
                upsampler=None if upsampler == None else upsampler,
                lambda_recon=150.0, 
                lambda_adv=0.25, 
                lambda_id=45, #original working lambda_id=45.0
                lambda_upsample=25, 
                lambda_sharp=0.0, 
                lambda_color=5.0,
                has_superres_generator=True if train_superres else False,
                name="aei_gan_model_with_superres" if train_superres else "aei_gan_model",
                use_emap=use_emap)
            model.compile(
                g_optimizer=AdamW(learning_rate=g_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5), # Previous weight_decay=1e-4 & learning_rate=4e-4
                d_optimizer=AdamW(learning_rate=d_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5), # Previous weight_decay=1e-4 & learning_rate=4e-6
                bce_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM), #For MULTI-GPU reduction=tf.keras.losses.Reduction.SUM is needed
            )
    else:
        model = AEIGANModel(
            generator=get_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_blocks=num_blocks, freeze=finetune_superres, freeze_all_except_deconv=deconv_only, with_super_resolution=train_superres),
            discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
            upsampler=None if upsampler == None else upsampler,
            lambda_recon=150.0, 
            lambda_adv=0.25, 
            lambda_id=45, #original working lambda_id=45.0
            lambda_upsample=25, 
            lambda_sharp=0.0, 
            lambda_color=5.0,
            has_superres_generator=True if train_superres else False,
            name="aei_gan_model_with_superres" if train_superres else "aei_gan_model",
            use_emap=use_emap)
        model.compile(
            g_optimizer=AdamW(learning_rate=g_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5), # Previous weight_decay=1e-4 & learning_rate=4e-4
            d_optimizer=AdamW(learning_rate=d_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5), # Previous weight_decay=1e-4 & learning_rate=4e-6
        )

    # Load old weights if required
    if model_weights and os.path.isdir(model_weights):
        print(f"###### Using model weights: {model_weights}")
        model.load_weights(model_weights)

        ###Temporary - START
        #new_upsampler_layer = model.generator.get_layer("z_attr8_upsampler")
        #print("######################################")
        #print(new_upsampler_layer.get_weights())
        #print("######################################")
        #input_shape_temp = (None, None, 32)  # This should match the input shape used in your upsampler_model
        #temp_input = tf.keras.Input(shape=input_shape_temp)
        #temp_output = SubPixelUpsampler(out_channels=32, scale=2, kernel_size=3, activation=None, name="z_attr8_upsampler")(temp_input)
        #temp_upsampler_model = tf.keras.Model(inputs=temp_input, outputs=temp_output)
        #temp_upsampler_model.load_weights("./models/upsampler_weights.h5")
        #trained_upsampler_weights = temp_upsampler_model.get_layer("z_attr8_upsampler").get_weights()
        #new_upsampler_layer.set_weights(trained_upsampler_weights)
        #print("######################################")
        #print(new_upsampler_layer.get_weights())
        #print("######################################")
        #new_upsampler_layer.trainable=False
        ###Temporary - END

        #Added only for the superres training
        #if train_superres or new_discriminator:
        #    model.discriminator=MinimalPatchGAN(ndf=64, num_layers=3)
        
        model.g_optimizer=AdamW(learning_rate=g_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5) # Previous weight_decay=1e-4 & learning_rate=4e-4
        model.d_optimizer=AdamW(learning_rate=d_learning_rate, beta_1=0, beta_2=0.999, weight_decay=1e-5) # Previous weight_decay=1e-4 & learning_rate=4e-6
    
    # Prime the models with fake data    
    dummy_img = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], dtype=tf.float16)
    dummy_embed = tf.zeros([1, NUM_FEATURES], dtype=tf.float16)
    model((dummy_img, dummy_embed), training=False)
    #if upsampler:
    #    upsampler(dummy_img)
    
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
        train, validation, tfrecord_path = get_training_data(
            batch_size=DATASET_BATCH_SIZE, 
            tfrecord_shard_path=tfrecord_shard_path, 
            p=0.01,
            shard_index=None if randomize_shards else i,
            shuffle_buffer_size=1000,
            validation_dataset=True
        ) 

        if max_batches is not None:
            train = train.take(max_batches)

        if validation:
            _, _, image_generator = get_callbacks(validation)
        else:
            _, _, image_generator = get_callbacks(train.take(1))

        #image_generator.steps_interval = 500

        #if train_superres:
        #    image_generator.steps_interval = 50

        # Epoch calculation incase we have uneven divisibility
        current_epochs = epochs_per_fit + (1 if i < remaining_epochs else 0)

        model.pre_fit_setup(os.path.basename(tfrecord_path))

        print("###### STARTING TRAINING")
        # aei-net training

        model.fit(
            train,
            #validation_data=validation,  # optional
            epochs=current_epochs,
            callbacks=[
                #FinalModelSaver(f"{model_save_path}_BLOCKS{NUM_BLOCKS}", current_epochs, save_epoch),
                terminate_on_nan,
                # model_checkpoint_callback,
                # tensorboard_callback,
                image_generator,
                #adaptive_id_loss_cb,
            ]
        )
        


    # Final Model Save
    model.save(f"{model_save_path}_BLOCKS{num_blocks}_latest")
    print(f"Model saved at: {model_save_path}_BLOCKS{num_blocks}_latest")

        
        
        
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
        #default="./models/MODEL_256x256_v2_BLOCKS2_latest",
        default="",
        help="Path to the model weights (default: './models/MODEL_256x256_v2_BLOCKS2_latest')"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models/aei_net_gan_weights",
        help="Path to the super resolution model weights (default: './models/aei_net_gan_weights')"
    )
    parser.add_argument(
        "--train_superres",
        action="store_true",
        default=False,
        help="Whether to train the super resolution model (default: False)"
    )
    parser.add_argument(
        "--new_discriminator",
        action="store_true",
        default=False,
        help="Whether to reset the discriminator (default: False)"
    )
    parser.add_argument(
        "--randomize_shards",
        action="store_true",
        default=False,
        help="Randomize training dataset selection (default: False)"
    )
    parser.add_argument(
        "--deconv_only",
        action="store_true",
        default=False,
        help="Freeze all but the upsampling blocks i.e. Deconv4x4 while training (default: False)"
    )
    parser.add_argument(
        "--finetune_superres",
        action="store_true",
        default=False,
        help="Superres block fine tuning (default: False)"
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=1,
        help="Interval of epochs to save the model (default: 1)"
    )
    parser.add_argument(
        "--tfrecord_shard_path",
        type=str,
        default="./data/kitchensink_256/",
        help="Path to the dataset (default: './data/kitchensink_256/')"
    )
    parser.add_argument(
        "--g_learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate of generator (default: 1e-4)"
    )
    parser.add_argument(
        "--d_learning_rate",
        type=float,
        default=0.000001,
        help="Learning rate of discriminator (default: 1e-6)"
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Enable multi-GPU training using tf.distribute.MirroredStrategy (default: False)"
    )
    parser.add_argument(
        "--use_emap",
        action="store_true",
        default=False,
        help="Enable legacy emap training (default: False)"
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        help="Number of AADGenerator Blocks (default: 2)"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum Number of Batches  (default: None)"
    )

    args = parser.parse_args()

    #if not args.train_superres:
    #tf.config.optimizer.set_jit(True)

    train(
        total_epochs=args.total_epochs,
        reload_interval=args.reload_interval,
        model_save_path=args.model_save_path,
        model_weights=args.model_weights,
        train_superres=args.train_superres,
        tfrecord_shard_path=args.tfrecord_shard_path,
        save_epoch=args.save_epoch,
        new_discriminator=args.new_discriminator,
        deconv_only=args.deconv_only,
        randomize_shards=args.randomize_shards,
        g_learning_rate=args.g_learning_rate,
        d_learning_rate=args.d_learning_rate,
        multi_gpu=args.multi_gpu,
        finetune_superres=args.finetune_superres,
        use_emap=args.use_emap,
        num_blocks=args.num_blocks,
        max_batches=args.max_batches
    )
