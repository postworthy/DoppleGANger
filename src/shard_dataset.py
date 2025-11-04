import tensorflow as tf
import random
import os

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

    # Reshape to the expected shapes (make sure the total number of elements matches)
    img_batch = tf.reshape(img_batch, img_shape)
    embed_batch = tf.reshape(embed_batch, embed_shape)
    y_target = tf.reshape(y_target, y_target_shape)
    y_target_x2 = y_target  # Adjust if needed

    return (img_batch, embed_batch), (y_target, y_target_x2)

def parse_and_resize(
    record,
    orig_batch_size=32,
    orig_height=256,
    orig_width=256,
    orig_channels=3,
    target_height=256,
    target_width=256
):
    """
    Parses a single serialized tf.train.Example containing:
      - 'img_batch': raw float16 image data
      - 'embed_batch': raw float16 embedding
      - 'Y_target': raw float16 target image data
    Reshapes and resizes the images, and then re-encodes them.
    """

    img_shape=(orig_batch_size, orig_width, orig_width, orig_channels)
    embed_shape=(orig_batch_size, 512)
    y_target_shape=(orig_batch_size, orig_width, orig_width, orig_channels)

    (img_batch, embed_batch), (y_target, y_target_x2) = parse_tfrecord_fn(record, img_shape, embed_shape, y_target_shape, parse_method="parse_tensor")

    # Resize the images; keep float16
    # (If your data is in [0,1], or [-1,1], watch out for changes in interpolation.)
    if orig_height != target_height and orig_width != target_width:
        img_batch_resized = tf.image.resize(img_batch, [target_height, target_width])
        y_target_resized = tf.image.resize(y_target, [target_height, target_width])
    else:
        img_batch_resized = img_batch
        y_target_resized = y_target

    # embed_batch is presumably not an image, so we don't resize that
    # If embed_batch has some known shape, you may still want to reshape it:
    # embed_batch = tf.reshape(embed_batch, [...])

    # Re-encode the resized images as raw float16
    img_batch_resized_raw = tf.io.serialize_tensor(img_batch_resized)
    y_target_resized_raw = tf.io.serialize_tensor(y_target_resized)

    # We also need to store the original embedding. We could either re-serialize
    #    the same embed_batch or keep it as is. Suppose we store it as a serialized tensor as well:
    embed_batch_serialized = tf.io.serialize_tensor(embed_batch)

    # Build a new Example with the resized data
    #    - The .numpy() calls are needed if you're running this in eager mode.
    #      If you're writing a tf.data pipeline, you might need a different approach
    #      (see note below).
    feature_dict = {
        'img_batch': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_batch_resized_raw.numpy()])
        ),
        'embed_batch': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embed_batch_serialized.numpy()])
        ),
        'Y_target': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[y_target_resized_raw.numpy()])
        )
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

def split_tfrecord_into_shards(
    input_tfrecord_path,
    output_prefix,
    num_shards=250,
    target_height=256,
    target_width=256,
    orig_batch_size=32,
):
    """
    Streams a large TFRecord, randomly distributes each record into ~num_shards shards,
    and writes to TFRecords with the pattern: 
      {output_prefix}_shard_0.tfrecord, ..., {output_prefix}_shard_(num_shards-1).tfrecord
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create TFRecord writers for each shard
    writers = [
        tf.io.TFRecordWriter(f"{output_prefix}_shard_{i}.tfrecord")
        for i in range(num_shards)
    ]

    # Use TFRecordDataset to stream the data
    dataset = tf.data.TFRecordDataset(input_tfrecord_path)

    # Randomly assign each record to a shard
    for record in dataset:
        resized_serialized_example = parse_and_resize(
            record, 
            target_height=target_height, 
            target_width=target_width,
            orig_batch_size=orig_batch_size,
        )

        shard_index = random.randrange(num_shards)
        writers[shard_index].write(resized_serialized_example)

    # Close all writers
    for w in writers:
        w.close()

if __name__ == "__main__":
    # Example usage
    input_file = "./data/kitchensink_256/combined.tfrecord"
    output_prefix = "./data/kitchensink_256/kitchensink_256"

    split_tfrecord_into_shards(
        input_tfrecord_path=input_file,
        output_prefix=output_prefix,
        num_shards=250,
        target_height=256,  # Modify if you want a different size
        target_width=256,
        orig_batch_size=1,
    )

