import argparse
from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.05)

import tensorflow as tf
import os
from onnx_upsampler import Upsampler_Remote
from data_loader import determine_parse_method, get_random_shard_tfrecord, parse_tfrecord_fn  # New data loader import
from tensorflow.keras import mixed_precision
from tqdm import tqdm

mixed_precision.set_global_policy('mixed_float16')

upsampler = Upsampler_Remote()

def parse_and_resize_with_dataloader(
    record,
    orig_batch_size=32,
    orig_height=128,
    orig_width=128,
    orig_channels=3,
    target_height=256,
    target_width=256,
    parse_method="parse_tensor",
    embed_shape=None
):
    """
    Uses the new dataloader function (parse_tfrecord_fn) to parse the record,
    then applies upsampling and resizing operations and casts outputs to float16.
    
    If embed_shape is not provided, it is assumed to be the same as the image shape.
    """
    if embed_shape is None:
        # Adjust this if your embed_batch tensor has a different shape
        embed_shape = [orig_batch_size, 512]
    
    # Define expected shapes for the tensors
    img_shape = [orig_batch_size, orig_height, orig_width, orig_channels]
    y_target_shape = [orig_batch_size, orig_height, orig_width, orig_channels]
    
    # Use the new dataloader function to parse and reshape the tensors
    (img_batch, embed_batch), (y_target, _) = parse_tfrecord_fn(
        record,
        img_shape=img_shape,
        embed_shape=embed_shape,
        y_target_shape=y_target_shape,
        parse_method=parse_method
    )
    
    
    return (img_batch, embed_batch, y_target)

def build_example_in_python(img_batch_tensor, embed_batch_tensor, y_target_tensor):
    """
    Builds a tf.train.Example from Tensors in Python eager mode,
    converting tensors to numpy arrays to avoid autograph issues.
    """
    img_bytes   = tf.io.serialize_tensor(img_batch_tensor).numpy()
    embed_bytes = tf.io.serialize_tensor(embed_batch_tensor).numpy()
    y_bytes     = tf.io.serialize_tensor(y_target_tensor).numpy()

    feature_dict = {
        'img_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        'embed_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embed_bytes])),
        'Y_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_bytes]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

def save_as_upsampled_tfrecord_fast(
    input_tfrecord_path,
    output_tfrecord_path,
    target_height=256,
    target_width=256,
    orig_batch_size=32,
    orig_height=128,
    orig_width=128,
    orig_channels=3,
    num_parallel_reads=4,
    embed_shape=None
):
    """
    Reads records from the input TFRecord file, applies parsing with the new dataloader
    function followed by upsampling/resizing operations, and writes out a new TFRecord.
    """
    # Create a dataset from the input TFRecord file
    dataset = tf.data.TFRecordDataset(
        [input_tfrecord_path],
        num_parallel_reads=num_parallel_reads
    )

    parse_method = determine_parse_method(dataset)

    print(f"##### Parsing with {parse_method}")

    # Parse and process records in parallel using our new helper function
    dataset = dataset.map(
        lambda x: parse_and_resize_with_dataloader(
            x,
            orig_batch_size=orig_batch_size,
            orig_height=orig_height,
            orig_width=orig_width,
            orig_channels=orig_channels,
            target_height=target_height,
            target_width=target_width,
            parse_method=parse_method,
            embed_shape=embed_shape
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch to improve performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Write the processed examples to a new TFRecord file
    writer = tf.io.TFRecordWriter(output_tfrecord_path)
    for (img_batch, embed_batch, y_target) in tqdm(dataset, desc="Processing TFRecords"):
        # Upsample using the onnx_upsampler
        #img_batch = onnx_upsample(img_batch)
        y_target = upsampler.onnx_upsample(y_target)
        
        # Resize images to the target dimensions
        img_batch_resized = tf.image.resize(img_batch, [target_height, target_width])
        y_target_resized  = tf.image.resize(y_target, [target_height, target_width])
        
        # Cast to float16 to reduce memory and I/O overhead
        img_batch_resized = tf.cast(img_batch_resized, tf.float16)
        y_target_resized  = tf.cast(y_target_resized, tf.float16)
        embed_batch       = tf.cast(embed_batch, tf.float16)

        example_str = build_example_in_python(img_batch_resized, embed_batch, y_target_resized)
        writer.write(example_str)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsample dataset")

    parser.add_argument(
        "--tfrecord_shard_path",
        type=str,
        default="./data/celeba_liveportrait_256/",
        help="Path to the dataset (default: './data/celeba_liveportrait_256/')"
    )
    parser.add_argument(
        "--tfrecord_shard_path_out",
        type=str,
        default="./data/celeba_liveportrait_256_upsampled/",
        help="Output path (default: './data/celeba_liveportrait_256_upsampled/')"
    )
    
    args = parser.parse_args()

    input_file = get_random_shard_tfrecord(args.tfrecord_shard_path, args.tfrecord_shard_path_out)
    output_file = os.path.join(f"{args.tfrecord_shard_path_out}", os.path.basename(input_file))

    print(f"#### Upsampling: {input_file}")
    print(f"#### Destination: {output_file}")

    save_as_upsampled_tfrecord_fast(
        input_tfrecord_path=input_file,
        output_tfrecord_path=output_file,
        target_height=256,
        target_width=256,
        orig_batch_size=32,
        orig_height=256,
        orig_width=256,
        orig_channels=3,
        num_parallel_reads=4  # Adjust this (e.g., 2, 4, 8, ...) for best performance
    )

    # Once verified, you can remove or rename the original file:
    # os.remove(input_file)
    # os.rename(output_file, input_file)
