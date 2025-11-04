# Limit TensorFlow to 80% of GPU memory
from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.35)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from face_analysis import FaceAnalysis
from inswapper import INSwapper
from face import Face
from datetime import datetime


IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
NUM_FEATURES = 512
Z_DIM = 200
LEARNING_RATE = 0.0005
EPOCHS = 1
BETA = 2000
LOAD_MODEL = True
TAKE_BATCHES = 500        
        
def serialize_example_old_raw(img_batch, embed_batch, y_target):
    """
    Convert the batch arrays into a tf.train.Example suitable for writing to TFRecord.
    We store each array as raw bytes via .tobytes().
    """
    # Flatten the arrays to raw bytes
    img_batch_bytes = img_batch.tobytes()
    embed_batch_bytes = embed_batch.tobytes()
    y_target_bytes = y_target.tobytes()

    # Create Features
    feature = {
        'img_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_batch_bytes])),
        'embed_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embed_batch_bytes])),
        'Y_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_target_bytes])),
    }

    # Build an Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def serialize_example(img_batch, embed_batch, y_target):
    """
    Convert the batch arrays into a tf.train.Example suitable for writing to TFRecord.
    We use tf.io.serialize_tensor(...) so that we can parse it later with tf.io.parse_tensor(...).
    """
    # Serialize each tensor
    img_batch_serialized = tf.io.serialize_tensor(img_batch)
    embed_batch_serialized = tf.io.serialize_tensor(embed_batch)
    y_target_serialized = tf.io.serialize_tensor(y_target)

    # Build Features
    feature = {
        'img_batch': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_batch_serialized.numpy()])
        ),
        'embed_batch': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embed_batch_serialized.numpy()])
        ),
        'Y_target': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[y_target_serialized.numpy()])
        ),
    }

    # Build an Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def do_inswapper_pretraining(
    data_dirs=["/app/data/celeba-dataset/img_align_celeba/img_align_celeba/"], 
    output_dir="./data/", 
    use_fixed_image=False, 
    fixed_img_from_path="/app/data/celeba-dataset/img_align_celeba/img_align_celeba/999999.jpg",
    shuffle=False,
    batch_size_override=None):
    
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    inswapper_destroy = not INSwapper.is_initialized()
    face_analyser_destroy = not FaceAnalysis.is_initialized()

    face_analyser = FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    inswapper = INSwapper('/root/.insightface/models/inswapper_128.onnx')
    emap = inswapper.emap

    # Ensure the output directory exists, or create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #else:
    #    # Check if directory is not empty
    #    if os.listdir(output_dir):
    #        print("Pretraining folder is not empty. Exiting to avoid overwriting.")
    #        return

    # Create a timestamp-based unique filename, e.g., training_YYYYMMDD_HHMMSS.tfrecord
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tfrecord_filename = f"training_{timestamp}_shard_complete.tfrecord"
    tfrecord_path = os.path.join(output_dir, tfrecord_filename)

    def load_and_preprocess_image(file_path):
        def safe_load(file_path):
            try:
                # Read and decode the image
                image = tf.io.read_file(file_path)
                image = tf.image.decode_jpeg(image, channels=3)  # Decode as JPEG
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.LANCZOS5)  # Resize
                return image
            except Exception as e:
                print(f"Error processing file {file_path.numpy().decode('utf-8')}: {e}")
                # Return an empty tensor to indicate a failed load
                return tf.zeros([0, 0, 0], dtype=tf.float32)

        # Use tf.py_function for custom Python logic
        return tf.py_function(
            func=safe_load, inp=[file_path], Tout=tf.float32
        )



    file_patterns = []
    for data_dir in data_dirs:
        file_patterns.append(os.path.join(data_dir, "**/*.jpg"))  # Recursively include .jpg files
        file_patterns.append(os.path.join(data_dir, "**/*.png"))  # Recursively include .png files
        file_patterns.append(os.path.join(data_dir, "*.jpg"))  # Recursively include .jpg files
        file_patterns.append(os.path.join(data_dir, "*.png"))  # Recursively include .png files


    list_ds = tf.data.Dataset.list_files(file_patterns, shuffle=shuffle)


    train_data = list_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.filter(lambda x: tf.reduce_prod(tf.shape(x)) > 0)  # Skip empty tensors

    train_data = train_data.batch(BATCH_SIZE if not batch_size_override else batch_size_override)
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Optionally load a fixed "from" image
    if use_fixed_image:
        fixed_img_from = load_and_preprocess_image(fixed_img_from_path)

    def get_target(img_into, img_from):
        def process_image(img_into_tensor, img_from_tensor):
            try:
                img_into_np = img_into_tensor.numpy()
                img_from_np = img_from_tensor.numpy()
                faces_into = face_analyser.get(img_into_np)
                faces_from = face_analyser.get(img_from_np)
                faces_into_sorted = sorted(faces_into, key=lambda x: x.bbox[0])
                faces_from_sorted = sorted(faces_from, key=lambda x: x.bbox[0])
                if faces_into_sorted and faces_from_sorted:
                    face_into = faces_into_sorted[0]
                    face_from = faces_from_sorted[0]
                    result = inswapper.get(img_into_np, face_into, face_from, paste_back=True)
                    embed = face_from.normed_embedding
                    embed = np.dot(embed, emap)
                    embed /= np.linalg.norm(embed)
                    return result.astype(np.float32), embed
                else:
                    # Fallback to random noise
                    noise = np.random.normal(loc=127.5, scale=50.0, size=img_into_np.shape)
                    noise = np.clip(noise, 0, 255).astype(np.uint8)
                    embed = np.random.normal(size=(NUM_FEATURES,)).astype(np.float32)
                    return noise.astype(np.float32), embed
            except Exception as e:
                print(f"Error while in process_image:\n{e}")
                noise = np.random.normal(loc=127.5, scale=50.0, size=img_into_np.shape)
                noise = np.clip(noise, 0, 255).astype(np.uint8)
                embed = np.random.normal(size=(NUM_FEATURES,)).astype(np.float32)
                return noise.astype(np.float32), embed

        border_size = 50
        img_into_padded = tf.pad(
            img_into,
            paddings=[[border_size, border_size], [border_size, border_size], [0, 0]],
            mode='CONSTANT',
            constant_values=255,
        )
        img_from_padded = tf.pad(
            img_from,
            paddings=[[border_size, border_size], [border_size, border_size], [0, 0]],
            mode='CONSTANT',
            constant_values=255,
        )
        Y_target_padded, embed = tf.py_function(
            func=process_image,
            inp=[img_into_padded, img_from_padded],
            Tout=(tf.float32, tf.float32)
        )
        Y_target = Y_target_padded[border_size:-border_size, border_size:-border_size, :]
        return Y_target, embed

    def prepare_inputs(img):
        img_processed = tf.cast(img, "float32") / 255.0
        shuffled_indices = tf.random.shuffle(tf.range(BATCH_SIZE if not batch_size_override else batch_size_override))
        img_random = tf.gather(img, shuffled_indices)
        indices = tf.range(BATCH_SIZE if not batch_size_override else batch_size_override)

        def get_target_pair(idx):
            img_i = img[idx]
            img_j = img_random[idx]
            if use_fixed_image:
                img_j = fixed_img_from
            return get_target(img_i, img_j)

        Y_target, embed = tf.map_fn(get_target_pair, indices, dtype=(tf.float32, tf.float32))
        Y_target.set_shape(img.shape)
        embed.set_shape([BATCH_SIZE if not batch_size_override else batch_size_override, NUM_FEATURES])  # shape depends on inswapper

        Y_target = tf.cast(Y_target, "float32") / 255.0
        return ((img_processed, embed), Y_target)

    # 1) Build the dataset pipeline
    train = train_data.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.apply(tf.data.experimental.ignore_errors())
    train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2) Create TFRecordWriter (single file)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        batch_index = 0
        for ((img_batch, embed_batch), Y_target) in train:
            # Convert to NumPy
            img_batch_np = img_batch.numpy().astype(np.float16)
            embed_batch_np = embed_batch.numpy().astype(np.float16)
            Y_target_np = Y_target.numpy().astype(np.float16)

            # Serialize each batch as a TFRecord Example
            example_bytes = serialize_example(img_batch_np, embed_batch_np, Y_target_np)
            writer.write(example_bytes)

            batch_index += 1

    #Clean up
    if inswapper_destroy:
        inswapper.destroy()
    if face_analyser_destroy:
        face_analyser.destroy()

    print(f"Done writing {batch_index} batches to TFRecord: {tfrecord_path}")
    return tfrecord_path

