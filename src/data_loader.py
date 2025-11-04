import os
import json
import numpy as np
import psutil
import tensorflow as tf
from augmentation import augment_batch
from pathlib import Path
import glob
import random
#from onnx_upsampler import Upsampler
IMAGE_SIZE=256

#upsampler=Upsampler()

def safe_reshape(tensor, target_shape):
    # Convert target_shape to a tensor and compute its product.
    target_shape_tensor = tf.convert_to_tensor(target_shape, dtype=tf.int32)
    expected_num_elements = tf.reduce_prod(target_shape_tensor)
    
    # Compute the number of elements in the input tensor.
    current_num_elements = tf.size(tensor)
    
    # Use TensorFlow's string formatting to generate a dynamic error message.
    error_msg = tf.strings.format(
        "Reshape failed: tensor has {} elements, but target shape requires {} elements.",
        (current_num_elements, expected_num_elements)
    )
    
    # Validate the element count.
    tf.debugging.assert_equal(
        current_num_elements,
        expected_num_elements,
        message=error_msg
    )
    
    # If the check passes, perform the reshape.
    return tf.reshape(tensor, target_shape)


def parse_tfrecord_fn(example_proto, img_shape, embed_shape, y_target_shape, parse_method="parse_tensor"):
    feature_description = {
        'img_batch': tf.io.FixedLenFeature([], tf.string),
        'embed_batch': tf.io.FixedLenFeature([], tf.string),
        'Y_target': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    if parse_method == "parse_tensor":
        img_batch = tf.io.parse_tensor(parsed_features["img_batch"], out_type=tf.float16)
        embed_batch = tf.io.parse_tensor(parsed_features["embed_batch"], out_type=tf.float16)
        y_target = tf.io.parse_tensor(parsed_features["Y_target"], out_type=tf.float16)
    elif parse_method == "decode_raw":
        img_batch = tf.io.decode_raw(parsed_features["img_batch"], tf.float16)
        embed_batch = tf.io.decode_raw(parsed_features["embed_batch"], tf.float16)
        y_target = tf.io.decode_raw(parsed_features["Y_target"], tf.float16)
    else:
        raise ValueError(f"Unsupported parse_method: {parse_method}")

    # Cast to float32 if desired
    #img_batch = tf.cast(img_batch, tf.float32)
    #embed_batch = tf.cast(embed_batch, tf.float32)
    #y_target = tf.cast(y_target, tf.float32)

    # Reshape to the expected shapes (make sure the total number of elements matches)
    
    img_batch = safe_reshape(img_batch, img_shape)
    embed_batch = safe_reshape(embed_batch, embed_shape)
    y_target = safe_reshape(y_target, y_target_shape)
    y_target_x2 = y_target  # Adjust if needed
    #y_target_x2 = upsampler.onnx_upsample(y_target)  # Adjust if needed

    if img_shape[0] == 1:
        img_batch = tf.squeeze(img_batch, axis=0)
        embed_batch = tf.squeeze(embed_batch, axis=0)
        y_target = tf.squeeze(y_target, axis=0)

    #print(img_batch.shape)
    #print(embed_batch.shape)
    #print(y_target.shape)

    return (img_batch, embed_batch), (y_target, y_target_x2)




def load_pretraining_dataset_tfrecord(tfrecord_path,
                                      img_shape=(32, IMAGE_SIZE, IMAGE_SIZE, 3),
                                      embed_shape=(32, 512),
                                      y_target_shape=(32, IMAGE_SIZE, IMAGE_SIZE, 3),
                                      batch_size=None,
                                      shuffle_buffer=0):
    """
    Loads the .tfrecord created above into a tf.data.Dataset.
    - If your data was stored as multiple 'batches' per file, adjust shapes accordingly.
    - If 'batch_size' is None, it implies your data is already in 'batch' form inside each record.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    parse_method = determine_parse_method(dataset)
    
    # Shuffle the records if desired
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    # Parse
    dataset = dataset.map(
        lambda x: parse_tfrecord_fn(x, img_shape, embed_shape, y_target_shape, parse_method),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    #dataset = dataset.cache()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_pretraining_dataset_tfrecord_with_augmentation(
    tfrecord_path,
    img_shape=(32, IMAGE_SIZE, IMAGE_SIZE, 3),
    embed_shape=(32, 512),
    y_target_shape=(32, IMAGE_SIZE, IMAGE_SIZE, 3),
    batch_size=None,
    shuffle_buffer=0,
    p=0.3
):
    """
    Loads the .tfrecord, parses each record into ((img_batch, embed_batch), Y_target),
    optionally shuffles, batches, and then applies synchronous augmentation 
    to (img_batch, Y_target) with probability p.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    parse_method = determine_parse_method(dataset)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)

    #print(f"img_shape:  {img_shape}")
    #print(f"embed_shape:  {embed_shape}")
    #print(f"y_target_shape:  {y_target_shape}")
    #print(f"parse_method:  {parse_method}")

    dataset = dataset.map(
        lambda x: parse_tfrecord_fn(x, img_shape, embed_shape, y_target_shape, parse_method),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    #print(next(iter(dataset)))

    #if batch_size is not None:
    #    dataset = dataset.batch(batch_size)


    def apply_augmentation(inputs, y_targets):
        """
        inputs = (img_batch, embed_batch)
        y_target = target images
        We'll augment (img_batch, y_target) in sync, leaving embed_batch alone.
        """
        img_batch, embed_batch = inputs
        (y_target, y_target_x2) = y_targets
        
        # Apply batch-level augmentation
        #BATCH_SIZE, H, W, C = y_target_x2.shape
        #img_batch_resized = tf.image.resize(img_batch, [H, W])
        #img_batch_aug, y_target_x2_aug = augment_batch(img_batch_resized, y_target_x2, p=p)
        #BATCH_SIZE, h, w, c = img_batch.shape
        #img_batch_aug_resized = tf.image.resize(img_batch_aug, [h, w])
        #y_target_aug = tf.image.resize(y_target_x2_aug, [h, w])
        img_batch_aug, y_target_aug = augment_batch(img_batch, y_target, p=p)
        
        #BATCH_SIZE, h, w, c = img_batch.shape
        #y_target_x2_aug = tf.image.resize(y_target_x2_aug, [h*2, w*2])
        y_target_x2_aug = y_target_aug # For now we dont need 2x so just assign it to be y_target

        # Return updated structure
        return (img_batch_aug, embed_batch), (y_target_aug, y_target_x2_aug)

    #dataset = dataset.cache()
    
    if p != 0:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_random_shard_tfrecord_OLD(root_dir):
    """
    Scans 'root_dir' for TFRecord files containing '_shard_' in the filename,
    randomly selects one, and returns its full absolute path.
    """
    pattern = os.path.join(root_dir, "*_shard_*.tfrecord")
    shard_files = glob.glob(pattern)
    
    if not shard_files:
        raise FileNotFoundError(f"No TFRecord shards found in directory: {root_dir}")
    
    selected_file = random.choice(shard_files)
    return os.path.abspath(selected_file)

def get_random_shard_tfrecord(root_dir, exclusion_dir=None):
    """
    Scans 'root_dir' for TFRecord files containing '_shard_' in the filename,
    optionally excludes files whose names also appear in 'exclusion_dir',
    randomly selects one, and returns its full absolute path.

    Args:
        root_dir (str): Directory to scan for TFRecord shard files.
        exclusion_dir (str, optional): Directory containing files whose matching names
                                       should be excluded from selection.

    Returns:
        str: The absolute path to the randomly selected TFRecord shard file.

    Raises:
        FileNotFoundError: If no valid TFRecord shard file is found in 'root_dir'.
    """
    # Gather all TFRecord shard files in the root directory.
    pattern = os.path.join(root_dir, "*_shard_*.tfrecord")
    shard_files = glob.glob(pattern)

    # If an exclusion directory is provided, filter out matching filenames.
    if exclusion_dir:
        exclusion_pattern = os.path.join(exclusion_dir, "*")
        excluded_names = {os.path.basename(f) for f in glob.glob(exclusion_pattern)}
        shard_files = [f for f in shard_files if os.path.basename(f) not in excluded_names]

    if not shard_files:
        msg = f"No TFRecord shards found in directory: {root_dir}"
        if exclusion_dir:
            msg += f" after excluding files from: {exclusion_dir}"
        raise FileNotFoundError(msg)

    selected_file = random.choice(shard_files)
    return os.path.abspath(selected_file)

def get_nth_shard_tfrecord(root_dir, index, exclusion_dir=None):
    """
    Scans 'root_dir' for TFRecord files containing '_shard_' in the filename,
    optionally excludes files whose names also appear in 'exclusion_dir',
    selects the file at the given index (using modulus for rollover), and returns
    its full absolute path.

    Args:
        root_dir (str): Directory to scan for TFRecord shard files.
        index (int): The index of the file to select. If the index is out of bounds,
                     the modulus operation is used to wrap around.
        exclusion_dir (str, optional): Directory containing files whose matching names
                                       should be excluded from selection.

    Returns:
        str: The absolute path to the selected TFRecord shard file.

    Raises:
        FileNotFoundError: If no valid TFRecord shard file is found in 'root_dir'.
    """
    def natural_key(s):
        import re
        # Split into text and number chunks
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]


    # Gather all TFRecord shard files in the root directory.
    pattern = os.path.join(root_dir, "*_shard_*.tfrecord")
    shard_files = glob.glob(pattern)
    shard_files.sort(key=natural_key)

    # If an exclusion directory is provided, filter out matching filenames.
    if exclusion_dir:
        exclusion_pattern = os.path.join(exclusion_dir, "*")
        excluded_names = {os.path.basename(f) for f in glob.glob(exclusion_pattern)}
        shard_files = [f for f in shard_files if os.path.basename(f) not in excluded_names]

    if not shard_files:
        msg = f"No TFRecord shards found in directory: {root_dir}"
        if exclusion_dir:
            msg += f" after excluding files from: {exclusion_dir}"
        raise FileNotFoundError(msg)

    # Calculate the modulus to roll over the index if necessary.
    mod_index = index % len(shard_files)
    selected_file = shard_files[mod_index]
    return os.path.abspath(selected_file)



def determine_parse_method(dataset, out_type=tf.float16):
    """
    Determine which parse method should be used for the TFRecordDataset.
    
    This function retrieves one record from the dataset, extracts the 'img_batch' field,
    and attempts to parse it using tf.io.parse_tensor. If successful, it returns "parse_tensor",
    meaning the data was serialized with tf.io.serialize_tensor. If an error occurs (e.g., a UnicodeDecodeError),
    it returns "decode_raw", indicating that the data was serialized with .tobytes().
    
    Parameters:
      dataset (tf.data.TFRecordDataset): The dataset created by tf.data.TFRecordDataset.
      out_type (tf.DType): The expected output type of the parsed tensor (default: tf.float16).
    
    Returns:
      str: Either "parse_tensor" or "decode_raw".
    """
    parse_method = "decode_raw"
    # Attempt to get one record from the dataset.
    for record in dataset.take(1):
        # Define a minimal feature description to extract one field.
        feature_description = {
            'img_batch': tf.io.FixedLenFeature([], tf.string)
        }
        try:
            # Parse the example to get the serialized 'img_batch'.
            parsed_example = tf.io.parse_single_example(record, feature_description)
            serialized_img = parsed_example['img_batch']
            # Try to parse with tf.io.parse_tensor.
            _ = tf.io.parse_tensor(serialized_img, out_type=out_type)
            # If successful, we conclude the data was serialized with tf.io.serialize_tensor.
            parse_method = "parse_tensor"
        except Exception as e:
            # If an error occurs, assume the data was serialized with .tobytes() and should use decode_raw.
            parse_method = "decode_raw"
    
    print(f"#### Parsing with `{parse_method}`")
    return parse_method


def get_training_data(batch_size=32, image_size=256, num_features=512, tfrecord_shard_path="./data/kitchensink_128/", p=0.15, shuffle_buffer_size=320, shard_index=None, validation_dataset=True):
    if shard_index == None:
        tfrecord_path = get_random_shard_tfrecord(tfrecord_shard_path)
    else:
        tfrecord_path = get_nth_shard_tfrecord(tfrecord_shard_path, shard_index)

    print(f"##### Loading data from randon shard: {tfrecord_path} with batch_size: {batch_size}, image_size: {image_size}x{image_size}, num_features: {num_features}")

    dataset = load_pretraining_dataset_tfrecord_with_augmentation(
         tfrecord_path=tfrecord_path,                              # location of data
         img_shape=(batch_size, image_size, image_size, 3),        # shape of 'img_batch' in each record
         embed_shape=(batch_size, num_features),                   # shape of 'embed_batch' 
         y_target_shape=(batch_size, image_size, image_size, 3),   # shape of 'y_target_shape' in each record
         p=p
    )
    #print(next(iter(dataset)).shape)    

    if batch_size == 1:        
        dataset = dataset.batch(28).prefetch(buffer_size=tf.data.AUTOTUNE)

    
    if validation_dataset:
        validation = dataset.take(5).prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        validation = None

    if shuffle_buffer_size > 0:
        dataset = dataset.skip(5 if validation_dataset else 0).shuffle(buffer_size=shuffle_buffer_size)
    else:
        dataset = dataset.skip(5 if validation_dataset else 0)
    train = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    ################
    #Load into cache
    #def monitor_memory():
    #    """
    #    Monitor memory usage and return True if usage is below the threshold.
    #    """
    #    memory_info = psutil.virtual_memory()
    #    return memory_info.percent < (0.75 * 100)
    #dataset = dataset.cache()
    #take_n = 0
    #for _ in dataset:
    #    take_n += 1
    #    if monitor_memory():
    #        take_n += 1
    #        pass
    #    else:
    #        break
    #print(f"Take {take_n}")
    #train = dataset.take(take_n - 100).prefetch(buffer_size=tf.data.AUTOTUNE)
    ################

    
    
    return (train, validation, tfrecord_path)

def get_self_training_data_with_augmentation(batch_size=32, image_size=IMAGE_SIZE, num_features=512, tfrecord_shard_path="./data/kitchensink_128/", p=0.15):
    tfrecord_path = get_random_shard_tfrecord(tfrecord_shard_path)
    print(f"##### Loading data from random shard: {tfrecord_path}")

    dataset = load_pretraining_dataset_tfrecord(
         tfrecord_path=tfrecord_path,                              # location of data
         img_shape=(batch_size, image_size, image_size, 3),        # shape of 'img_batch' in each record
         embed_shape=(batch_size, num_features),                   # shape of 'embed_batch' 
         y_target_shape=(batch_size, image_size, image_size, 3),   # shape of 'y_target_shape' in each record
    )

    # Separate validation and training data if needed
    validation = dataset.take(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.skip(100)

    # Extract only the image batch from each record.
    train = dataset.map(lambda x, y: x[0], num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch so that each element is an individual image.
    train = train.unbatch()
    # Shuffle the individual images.
    train = train.shuffle(buffer_size=100000)
    
    # Create pairs of images by batching 2 images at a time (ensuring complete pairs).
    pairs = train.batch(2, drop_remainder=True)
    
    # Now, map each pair into a tuple: (first_image, second_image)
    paired = pairs.map(lambda pair: (pair[0], pair[1]), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Then batch these pairs into batches of size batch_size.
    paired_batches = paired.batch(batch_size, drop_remainder=True)

    # Apply augmentation to each batch of paired images.
    augmented_batches = paired_batches.map(lambda img_batch, y_batch: augment_batch(img_batch, y_batch, p=p),
                                             num_parallel_calls=tf.data.AUTOTUNE)
    
    return augmented_batches

