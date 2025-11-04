import tensorflow as tf
import argparse

def parse_tfrecord_fn(example_proto, img_shape, embed_shape, y_target_shape, parse_method="parse_tensor"):
    """
    Parse a TFRecord example using the specified parse_method.
    """
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

    # Cast to float16 and reshape to the expected shapes.
    img_batch = tf.cast(img_batch, tf.float16)
    embed_batch = tf.cast(embed_batch, tf.float16)
    y_target = tf.cast(y_target, tf.float16)

    img_batch = tf.reshape(img_batch, img_shape)
    embed_batch = tf.reshape(embed_batch, embed_shape)
    y_target = tf.reshape(y_target, y_target_shape)
    y_target_x2 = y_target  # Duplicate if needed

    return (img_batch, embed_batch), (y_target, y_target_x2)

def determine_parse_method(dataset, out_type=tf.float16):
    """
    Determine whether to use tf.io.parse_tensor or tf.io.decode_raw by testing one record.
    """
    parse_method = "decode_raw"
    for record in dataset.take(1):
        feature_description = {
            'img_batch': tf.io.FixedLenFeature([], tf.string)
        }
        try:
            parsed_example = tf.io.parse_single_example(record, feature_description)
            _ = tf.io.parse_tensor(parsed_example['img_batch'], out_type=out_type)
            parse_method = "parse_tensor"
        except Exception as e:
            parse_method = "decode_raw"
    print(f"#### Determined parse method: `{parse_method}`")
    return parse_method

def serialize_example(parsed_data):
    """
    Reserialize parsed tensors using tf.io.serialize_tensor so that the output TFRecord
    is stored in a parse_tensor–friendly format.
    """
    (img_batch, embed_batch), (y_target, _) = parsed_data

    img_bytes = tf.io.serialize_tensor(img_batch).numpy()
    embed_bytes = tf.io.serialize_tensor(embed_batch).numpy()
    y_target_bytes = tf.io.serialize_tensor(y_target).numpy()

    feature = {
        "img_batch": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        "embed_batch": tf.train.Feature(bytes_list=tf.train.BytesList(value=[embed_bytes])),
        "Y_target": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_target_bytes])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def serialize_example_split(parsed_data):
    (img_batch, embed_batch), (y_target, _) = parsed_data
    num_splits=32
    img_batches = tf.split(img_batch, num_splits, axis=0, num=None, name='split')
    embed_batches = tf.split(embed_batch, num_splits, axis=0, num=None, name='split')
    y_targets = tf.split(y_target, num_splits, axis=0, num=None, name='split')

    output=[]
    for i in range(num_splits):
        img_bytes = tf.io.serialize_tensor(img_batches[i]).numpy()
        embed_bytes = tf.io.serialize_tensor(embed_batches[i]).numpy()
        y_target_bytes = tf.io.serialize_tensor(y_targets[i]).numpy()

        feature = {
            "img_batch": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            "embed_batch": tf.train.Feature(bytes_list=tf.train.BytesList(value=[embed_bytes])),
            "Y_target": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_target_bytes])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        output.append(example.SerializeToString())

    return output

def process_record(raw_record, img_shape, embed_shape, y_target_shape, parse_method):
    """
    Process a single record:
      1. Parse using the specified parse_method.
      2. Reserialize it with tf.io.serialize_tensor.
    Returns a serialized TFRecord (as a Python bytes string).
    """
    parsed_data = parse_tfrecord_fn(raw_record, img_shape, embed_shape, y_target_shape, parse_method)
    serialized = serialize_example(parsed_data)
    return serialized

def process_record_split(raw_record, img_shape, embed_shape, y_target_shape, parse_method):
    """
    Process a single record:
      1. Parse using the specified parse_method.
      2. Reserialize it with tf.io.serialize_tensor.
    Returns a serialized TFRecord (as a Python bytes string).
    """
    parsed_data = parse_tfrecord_fn(raw_record, img_shape, embed_shape, y_target_shape, parse_method)
    serialized = serialize_example_split(parsed_data)
    return serialized

def record_generator(input_files, img_shape, embed_shape, y_target_shape):
    """
    Generator that loops over each input file. For each file, determine its parse method,
    then yield serialized records (i.e. parse_tensor–friendly) one by one.
    """
    for file in input_files:
        print(f"Processing file: {file}")
        # Create a dataset from the current file and determine its parse method.
        sample_ds = tf.data.TFRecordDataset(file)
        parse_method = determine_parse_method(sample_ds.take(1), out_type=tf.float16)
        # Re-open the dataset for actual iteration.
        ds = tf.data.TFRecordDataset(file)
        for raw_record in ds:
            try:
                serialized = process_record(raw_record, img_shape, embed_shape, y_target_shape, parse_method)
                yield serialized
            except Exception as e:
                print(f"Error processing record in {file}: {e}")
                continue

def record_generator_split(input_files, img_shape, embed_shape, y_target_shape):
    """
    Generator that loops over each input file. For each file, determine its parse method,
    then yield serialized records (i.e. parse_tensor–friendly) one by one.
    """
    for file in input_files:
        print(f"Processing file: {file}")
        # Create a dataset from the current file and determine its parse method.
        sample_ds = tf.data.TFRecordDataset(file)
        parse_method = determine_parse_method(sample_ds.take(1), out_type=tf.float16)
        # Re-open the dataset for actual iteration.
        ds = tf.data.TFRecordDataset(file)
        for raw_record in ds:
            try:
                serialized_array = process_record_split(raw_record, img_shape, embed_shape, y_target_shape, parse_method)
                for serialized in serialized_array:
                    yield serialized
            except Exception as e:
                print(f"Error processing record in {file}: {e}")
                continue

def combine_tfrecords_shuffled_streaming(output_file, input_files, img_shape, embed_shape, y_target_shape, buffer_size):
    """
    Combine multiple TFRecord files into one output file.
    Records are streamed via a generator that processes each file individually
    (with its own parse method) and a streaming shuffle is applied with a fixed buffer.
    """
    # Create a dataset from the generator.
    ds = tf.data.Dataset.from_generator(
        lambda: record_generator(input_files, img_shape, embed_shape, y_target_shape),
        output_types=tf.string,
        output_shapes=()
    )
    # Apply a streaming shuffle with a fixed buffer size.
    ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=False)

    writer = tf.io.TFRecordWriter(output_file)
    for serialized_record in ds:
        writer.write(serialized_record.numpy())
    writer.close()
    print(f"Combined and shuffled TFRecord file saved to: {output_file}")

def combine_tfrecords_shuffled_streaming_split(output_file, input_files, img_shape, embed_shape, y_target_shape, buffer_size):
    """
    Combine multiple TFRecord files into one output file.
    Records are streamed via a generator that processes each file individually
    (with its own parse method) and a streaming shuffle is applied with a fixed buffer.
    """
    # Create a dataset from the generator.
    ds = tf.data.Dataset.from_generator(
        lambda: record_generator_split(input_files, img_shape, embed_shape, y_target_shape),
        output_types=tf.string,
        output_shapes=()
    )
    # Apply a streaming shuffle with a fixed buffer size.
    ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=False)

    writer = tf.io.TFRecordWriter(output_file)
    for serialized_record in ds:
        writer.write(serialized_record.numpy())
    writer.close()
    print(f"Combined and shuffled TFRecord file saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./output/kitchensink_256/combined_shard_0.tfrecord",
                        help="Path to the output combined and shuffled TFRecord file")
    parser.add_argument("--buffer_size", type=int, default=100,
                        help="Shuffle buffer size for streaming shuffle")
    args = parser.parse_args()

    # List of input TFRecord files to combine.
    input_files = []
    for i in range(250):
        input_files.append(f"./data/kitchensink_256/kitchensink_256_shard_{i}.tfrecord")

    # Define expected tensor shapes. Adjust these as needed.
    IMG_SHAPE = (32, 256, 256, 3)
    EMBED_SHAPE = (32, 512)      # Update if needed.
    Y_TARGET_SHAPE = (32, 256, 256, 3)

    combine_tfrecords_shuffled_streaming_split(args.output, input_files, IMG_SHAPE, EMBED_SHAPE, Y_TARGET_SHAPE, args.buffer_size)

if __name__ == "__main__":
    main()
