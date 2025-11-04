#!/usr/bin/env python3
import math
import os
import glob
from gpu_memory import limit_gpu_memory
from aei_net import AEI_Net256
from AEIGANModel import AEIGANModel

# Limit GPU memory growth to avoid OOM
limit_gpu_memory(0.85)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from inference import AEINETSwapper
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from data_loader import get_training_data
from training_callbacks import get_callbacks
from tqdm import tqdm
from sr_model import FaceSuperResolutionModel
from discriminator import MinimalPatchGAN
from onnx_upsampler import Upsampler_Remote
mixed_precision.set_global_policy("mixed_float16")

DATASET_BATCH_SIZE = 32
DEFAULT_MODEL = "./models/MODEL_256x256_v18_BLOCKS2_latest"

import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import image as keras_image


class ImageGenerator(Callback):
    """
    Keras callback that periodically grabs a batch from `dataset`,
    runs super-resolution on it, and saves a composite of:
      [LR upscaled → SR prediction → HR target]
    side by side for `num_img` examples.
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        num_img: int = 5,
        steps_interval: int = 1000,
        output_dir: str = "./sr_output",
    ):
        super().__init__()
        self.dataset = dataset
        self.num_img = num_img
        self.steps_interval = steps_interval
        self.output_dir = output_dir
        self.total_steps = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self.total_steps += 1
        if self.total_steps % self.steps_interval == 0:
            try:
                self._save_composite(f"step_{self.total_steps}")
            except Exception as e:
                print(
                    f"[ImageGenerator] Error saving images at step {self.total_steps}: {e}"
                )

    def on_epoch_end(self, epoch, logs=None):
        try:
            self._save_composite(f"epoch_{epoch:03d}")
        except Exception as e:
            print(f"[ImageGenerator] Error saving images at epoch {epoch}: {e}")

    def _save_composite(self, tag: str):
        # Grab one batch from the dataset
        iterator = iter(self.dataset)

        (_, _), (lr_batch, _) = next(iterator)
        hr_batch = self.model(lr_batch, training=False)
        # Take first num_img examples
        lr = lr_batch[: self.num_img]
        hr = hr_batch[: self.num_img]

        # Upscale LR for visualization
        lr_up = tf.image.resize(lr, [512, 512], method="bicubic")

        # Convert tensors to numpy, clip to [0,1]
        lr_np = tf.clip_by_value(lr_up, 0.0, 1.0).numpy()
        hr_np = tf.clip_by_value(hr, 0.0, 1.0).numpy()

        # Build composite image: for each example, stack vertically: [lr; sr; hr]
        cols = []
        for i in range(self.num_img):
            col = np.vstack([lr_np[i], hr_np[i]])
            cols.append(col)

        # Then stack all columns horizontally
        composite = np.hstack(cols)
        # Convert to uint8
        composite = (composite * 255).astype(np.uint8)

        # Save
        fname = f"sr_{tag}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.join(self.output_dir, fname)
        keras_image.array_to_img(composite).save(path)
        print(f"[ImageGenerator] Saved composite to: {path}")


def train(args):
    os.makedirs(args.model_save_path, exist_ok=True)

    # Initialize and compile the super-resolution model
    aei = AEIGANModel(
                generator=AEI_Net256((256, 256, 3), c_id=512, num_blocks=2),
                discriminator=MinimalPatchGAN(ndf=64, num_layers=3),
                lambda_recon=150.0, lambda_adv=0.5, lambda_id=45.0)
    aei.load_weights(DEFAULT_MODEL)
    
    model = FaceSuperResolutionModel(
        upsampler=Upsampler_Remote(),
        face_swapper=aei.generator)
    model.build((None, 256, 256, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]

    )
    model.load_weights(os.path.join(args.model_save_path, "sr_model_final"))
    
    num_outer = math.ceil(args.total_epochs / args.reload_interval)
    epochs_per_fit = args.total_epochs // num_outer
    remaining_epochs = args.total_epochs - (epochs_per_fit * num_outer)

    for i in tqdm(range(num_outer), desc="Outer Loop Progress"):
        # Get Training and validation data
        train, validation, tfrecord_path = get_training_data(
            batch_size=DATASET_BATCH_SIZE,
            tfrecord_shard_path=args.tfrecord_shard_path,
            p=0.15,
            shard_index=i,
            shuffle_buffer_size=320,
            validation_dataset=True,
        )

        # Set up callbacks
        
        checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(args.model_save_path, "sr_epoch_{epoch:03d}.h5"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )
        earlystop_cb = EarlyStopping(
            monitor="val_loss", patience=5, mode="min", restore_best_weights=True
        )

        image_cb = ImageGenerator(
            dataset=validation, num_img=5, steps_interval=100, output_dir="./output"
        )

        # Epoch calculation incase we have uneven divisibility
        current_epochs = epochs_per_fit + (1 if i < remaining_epochs else 0)

        # Train
        model.fit(
            train,
            #validation_data=validation,
            epochs=current_epochs,
            callbacks=[checkpoint_cb, earlystop_cb, image_cb],
        )

    # Save final model
    model.save(os.path.join(args.model_save_path, "sr_model_final"))
    print(f"Model saved to {args.model_save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Face Super-Resolution Model")
    parser.add_argument(
        "--tfrecord_shard_path",
        type=str,
        default="./data/kitchensink_256/",
        help="Path to the dataset (default: './data/kitchensink_256/')"
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=250,
        help="Total number of epochs for training (default: 1)",
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=1,
        help="Interval for reloading the dataset (default: 1)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models/sr_model",
        help="Directory to save trained models",
    )
    args = parser.parse_args()

    train(args)
