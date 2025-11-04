import uuid
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing import image as keras_image

def resize_to_256(images):
    resized_images = []
    for img in images:
        # Check the shape of the image (height, width)
        if img.shape[0] != 256 or img.shape[1] != 256:
            # Resize the image to 128x128 using bilinear interpolation
            img_resized = tf.image.resize(img, [256, 256], method='bilinear').numpy()
        else:
            # Keep the image as is
            img_resized = img
        
        resized_images.append(img_resized)
    
    # Stack the resized images back into a batch
    return np.stack(resized_images)


def get_callbacks(train_dataset):
    # Create a model save checkpoint
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./checkpoint",
        save_weights_only=False,
        save_freq="epoch",
        monitor="g_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    tensorboard_callback = callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq = 1,
        profile_batch = '50,100')
                
    class ImageGenerator(callbacks.Callback):
        def __init__(self, dataset, num_img=10, steps_interval=4000, output_dir="./output"):
            super(ImageGenerator, self).__init__()
            self.dataset = dataset
            self.num_img = num_img
            self.steps_interval = steps_interval
            self.total_steps = 0
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        def on_train_batch_end(self, batch, logs=None):
            """
            Keras calls this after each training batch. We'll track how many steps
            have completed overall and save images every N steps.
            """
            self.total_steps += 1
            if self.total_steps % self.steps_interval == 0:
                try:
                    self.save_images(f"step_{self.total_steps}")
                except Exception as e:
                    print(f"{e}")


        def save_images(self, id):
            # Collect images, embeddings, and targets
            img_batch_list = []
            embed_batch_list = []
            y_target_list = []

            # Create an iterator over the dataset
            dataset_iterator = iter(self.dataset)
            #for _ in range(self.num_img):
            data = next(dataset_iterator)
            # We can handle 2 types of training
            # 1) Complete prepreocessed of shape => (img_batch, embed_batch), (y_target, y_target_x2)
            # 2) No preprocessing  of shape => (img_batch, extract_embeds_from)
            # If type #2 then we will perform embedding extraction and use self.teacher to get y_target and y_target_x2
            if len(data) == 2:
                if isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                    (img_batch, embed_batch), (y_target, y_target_x2) = data
                else:
                    img_batch, extract_embeds_from = data
                    embed_batch = self.model.extract_normed_embed(extract_embeds_from)
                    if self.model.teacher == None:
                        raise ValueError("Teacher model can't be none for this dataset shape!")
                    y_target = self.model.teacher([img_batch, embed_batch], training=False)
                    y_target_x2 = None            
            else:
                raise ValueError(f"Unexpected data structure length: {len(data)}")
            # Sample num_img images from the dataset
            
            img_batch_list.append(img_batch)
            embed_batch_list.append(embed_batch)
            y_target_list.append(y_target)

            # Stack images, embeddings, and targets
            img_batch = tf.concat(img_batch_list, axis=0)
            embed_batch = tf.concat(embed_batch_list, axis=0)
            y_target = tf.concat(y_target_list, axis=0)

            img_batch = tf.image.resize(img_batch, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR)

            # Generate predictions
            pred = self.model([img_batch, embed_batch], training=False)

            if isinstance(pred, tuple):
                y_pred, _ = pred
            else:
                y_pred = pred

            # Rescale images from [-1, 1] to [0, 1] if using 'tanh' activation
            y_pred = (y_pred + 1.0) / 2.0
            img_batch = (img_batch + 1.0) / 2.0
            y_target = (y_target + 1.0) / 2.0

            # Convert to NumPy
            img_batch_np = resize_to_256(img_batch.numpy())
            y_pred_np = resize_to_256(y_pred.numpy())
            y_target_np = resize_to_256(y_target.numpy())

            # Create a wide composite image
            columns = []
            for i in range(self.num_img):
                column = np.concatenate((
                    img_batch_np[i],  # Original image
                    y_pred_np[i],  # Generated image
                    y_target_np[i]  # Target image
                ), axis=0)  # Stack vertically for each example
                columns.append(column)

            # Stack all columns horizontally to create the wide image
            composite_image = np.concatenate(columns, axis=1)

            # Convert to uint8
            composite_image = np.clip(composite_image * 255, 0, 255).astype(np.uint8)

            # Save composite image
            random_suffix = uuid.uuid4().hex[:6]
            output_path = os.path.join(self.output_dir, f"composite_{id}_{random_suffix}.png")
            keras_image.array_to_img(composite_image).save(output_path)
            print(f"Composite image saved to {output_path}")


        def on_epoch_end(self, epoch, logs=None):
            self.save_images(f"epoch_{epoch:03d}")
            

    image_generator = ImageGenerator(dataset=train_dataset, num_img=10)

    return (model_checkpoint_callback, tensorboard_callback, image_generator)