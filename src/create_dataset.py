import os
import json
import numpy as np
import tensorflow as tf

def json_to_tfrecord(json_dir, tfrecord_path):
    """
    Reads all *.json files in `json_dir`, converts them to a TFRecord file at `tfrecord_path`.
    Each JSON file is expected to have keys:
        {
            "img_batch": [...],
            "embed_batch": [...],
            "Y_target": [...]
        }
    stored in some array-like structure.
    """
    
    # Use glob or os.scandir for more efficient listing
    json_files = tf.io.gfile.glob(os.path.join(json_dir, '*.json'))
    json_files.sort()  # optional, for consistency

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file_path in json_files:
            print(f"Loading: {file_path}")
            with tf.io.gfile.GFile(file_path, 'r') as f:
                data_dict = json.load(f)

            # Convert to numpy arrays
            img_batch = np.array(data_dict["img_batch"], dtype=np.float32)
            embed_batch = np.array(data_dict["embed_batch"], dtype=np.float32)
            Y_target = np.array(data_dict["Y_target"], dtype=np.float32)

            # Serialize each "batch" into a TFExample
            # (In your original code, each JSON file = 1 "batch". If you want,
            # you can break them into multiple records or keep them as single records.)
            example = _create_tfexample(img_batch, embed_batch, Y_target)
            writer.write(example.SerializeToString())

def _create_tfexample(img_batch, embed_batch, Y_target):
    """
    Convert our numpy arrays into a TFExample suitable for TFRecord.
    """
    feature = {
        'img_batch': _bytes_feature(img_batch.tobytes()),
        'embed_batch': _bytes_feature(embed_batch.tobytes()),
        'Y_target': _bytes_feature(Y_target.tobytes()),
        # We might also store shapes if they vary, e.g., height/width,
        # but here we assume consistent shapes or we handle shaping later.
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(json_dir="./data/pretraining/", tfrecord_path = "./data/training.tfrecord"):
    # Example usage:
    json_to_tfrecord(json_dir, tfrecord_path)
    print(f"Done creating {tfrecord_path}")
    
if __name__ == "__main__":
    main()
