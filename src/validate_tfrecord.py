#!/usr/bin/env python
import tensorflow as tf
import argparse

def parse_example(serialized_example):
    # Define the expected features; they are stored as raw bytes.
    feature_description = {
        'img_batch': tf.io.FixedLenFeature([], tf.string),
        'embed_batch': tf.io.FixedLenFeature([], tf.string),
        'Y_target': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(serialized_example, feature_description)

def process_first_records(tfrecord_file, num_records=25):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    count = 0
    for raw_record in dataset.take(num_records):
        try:
            # Parse the raw record into its features.
            example = parse_example(raw_record)
            # Decode the serialized tensors.
            img_tensor = tf.io.parse_tensor(example['img_batch'], out_type=tf.float16)
            embed_tensor = tf.io.parse_tensor(example['embed_batch'], out_type=tf.float16)
            y_tensor = tf.io.parse_tensor(example['Y_target'], out_type=tf.float16)
            
            # Print the shapes for this record.
            print(f"Record {count+1}:")
            print("  img_batch shape:", img_tensor.shape)
            print("  embed_batch shape:", embed_tensor.shape)
            print("  Y_target shape:", y_tensor.shape)
            count += 1
        except Exception as e:
            print(f"Error parsing record {count+1}: {e}")
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process first 25 records in a TFRecord file and print shapes.'
    )
    parser.add_argument(
        '--tfrecord',
        type=str,
        required=True,
        help='Path to the TFRecord file to process.'
    )
    args = parser.parse_args()
    
    processed = process_first_records(args.tfrecord, num_records=25)
    print(f"Processed {processed} records.")
