from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.75)

import argparse
from data_loader import get_random_shard_tfrecord, load_pretraining_dataset_tfrecord, safe_reshape
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
from emap import emap
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

class DatasetAnalyzer:
    def __init__(self):
        self.arcface = load_model("ArcFace-Res50.h5", compile=False)
        self.emap_tf = tf.convert_to_tensor(emap)
        with tf.device('/CPU:0'):
            self.emap_inv_tf = tf.linalg.inv(tf.convert_to_tensor(emap))

    def extract_embed(self, y_pred):
        try:
            resized_batch = tf.image.resize(y_pred, size=(112, 112))
            if len(resized_batch.shape) == 3:
                resized_batch = tf.expand_dims(resized_batch, axis=0)
            embed = self.arcface(resized_batch)
            return embed
        except Exception as e:
            tf.print(f"Error during embedding extraction: {e}")
            tf.print(f"y_pred.shape: {y_pred.shape}")
            batch_size = tf.shape(y_pred)[0]
            embed_shape = (batch_size, 512)
            noise = tf.random.normal(embed_shape, mean=0.0, stddev=1.0)
            return noise

    def extract_normed_embed(self, y_pred):
        latent = self.extract_embed(y_pred)
        latent = tf.math.l2_normalize(latent, axis=1)
        latent = tf.linalg.matmul(latent, self.emap_tf)
        latent = tf.math.l2_normalize(latent, axis=1)
        return latent
    
    @tf.function(jit_compile=True)
    def calculate_similarity(self, y_target, embed_true):
        embed_pred = self.extract_normed_embed(y_target)
        embed_pred = tf.cast(embed_pred, tf.float16)
        #embed_true = tf.linalg.matmul(embed_true, self.emap_inv_tf)
        #embed_true = tf.nn.l2_normalize(embed_true, axis=1)

        # Ensure embeddings are cast to float32 before computation
        #embed_true = tf.cast(embed_true, tf.float32)
        #embed_pred = tf.cast(embed_pred, tf.float32)

        embed_true_normalized = tf.nn.l2_normalize(embed_true, axis=1)
        embed_pred_normalized = tf.nn.l2_normalize(embed_pred, axis=1)

        cosine_similarity = tf.reduce_sum(embed_true_normalized * embed_pred_normalized, axis=1)

        #tf.print("Raw Similarity Values:", cosine_similarity)

        # Explicitly clip values to prevent numerical drift issues
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)

        return (1 - cosine_similarity) / 2




    @tf.function(jit_compile=True)
    def tf_median(self, tensor):
        tensor_sorted = tf.sort(tensor)
        n = tf.size(tensor_sorted)
        mid = n // 2
        median = tf.cond(
            n % 2 == 0,
            lambda: (tensor_sorted[mid - 1] + tensor_sorted[mid]) / 2.0,
            lambda: tensor_sorted[mid]
        )
        return median
    
    def analyze_dataset(self, dataset, progress_every_n=100, batch_size=128):
        #total_batches = dataset.reduce(0, lambda x, _: x + 1).numpy()
        #print(f"Total dataset batches: {total_batches}")

        embed_similarities = tf.TensorArray(dtype=tf.float16, size=0, dynamic_size=True)
        idx = 0

        progress_bar = tqdm(dataset, desc="Analyzing dataset")
        for i, (_, embed_batch, y_target) in enumerate(progress_bar):
            similarity = self.calculate_similarity(y_target, embed_batch)
            embed_similarities = embed_similarities.write(idx, similarity)
            idx += 1

            #tf.print("similarity:", similarity)

            if (i + 1) % progress_every_n == 0:
                current_embed_similarities = tf.reshape(embed_similarities.stack(), [-1])
                mean_diff = tf.reduce_mean(current_embed_similarities)
                median_diff = self.tf_median(current_embed_similarities)
                std_diff = tf.math.reduce_std(current_embed_similarities)

                current_embed_similarities_np = current_embed_similarities.numpy()
                percentiles = [25, 75, 90, 95, 99]
                percentile_values = np.percentile(current_embed_similarities_np, percentiles)

                progress_bar.set_postfix({
                    "Mean": f"{mean_diff.numpy():.4f}",
                    "Median": f"{median_diff.numpy():.4f}",
                    "Std": f"{std_diff.numpy():.4f}",
                    "P25": f"{percentile_values[0]:.4f}",
                    "P75": f"{percentile_values[1]:.4f}",
                    "P90": f"{percentile_values[2]:.4f}",
                    "P95": f"{percentile_values[3]:.4f}",
                    "P99": f"{percentile_values[4]:.4f}",
                }, refresh=True)

        embed_similarities = embed_similarities.concat()

        mean_diff = tf.reduce_mean(embed_similarities)
        median_diff = self.tf_median(embed_similarities)
        std_diff = tf.math.reduce_std(embed_similarities)

        embed_similarities_np = embed_similarities.numpy()
        percentiles = [25, 75, 90, 95, 99]
        percentile_values = np.percentile(embed_similarities_np, percentiles)

        suggested_cutoff = 1 - percentile_values[1]  # Using 95th percentile as recommended cutoff

        tf.print("Final Mean:", mean_diff)
        tf.print("Final Median:", median_diff)
        tf.print("Final Std Dev:", std_diff)
        for p, val in zip(percentiles, percentile_values):
            tf.print(f"P{p}:", val)

        tf.print("Suggested optimal cutoff:", suggested_cutoff)
        return suggested_cutoff



def _parse_function(example_proto, img_shape, embed_shape, y_target_shape):
    feature_description = {
        'img_batch': tf.io.FixedLenFeature([], tf.string),
        'embed_batch': tf.io.FixedLenFeature([], tf.string),
        'Y_target': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    img_batch = tf.io.parse_tensor(parsed_example['img_batch'], tf.float16)
    embed_batch = tf.io.parse_tensor(parsed_example['embed_batch'], tf.float16)
    y_target = tf.io.parse_tensor(parsed_example['Y_target'], tf.float16)
    
    img_batch = safe_reshape(img_batch, img_shape)
    embed_batch = safe_reshape(embed_batch, embed_shape)
    y_target = safe_reshape(y_target, y_target_shape)

    if img_shape[0] == 1:
        img_batch = tf.squeeze(img_batch, axis=0)
        embed_batch = tf.squeeze(embed_batch, axis=0)
        y_target = tf.squeeze(y_target, axis=0)

    return img_batch, embed_batch, y_target

def filter_dataset(dataset, output_tfrecord, arcface_analyzer, median=0.4995, std=0.0012):
    writer = tf.io.TFRecordWriter(output_tfrecord)

    total_written = 0
    total_excluded = 0

    progress_bar = tqdm(dataset, desc="Filtering dataset")

    for img_batch, embed_batch, y_target in progress_bar:
        similarity = arcface_analyzer.calculate_similarity(y_target, embed_batch)

        # Ensure similarity has the correct batch shape (batch_size,)
        similarity = tf.reshape(similarity, [-1])  

        # Ensure mask is a boolean tensor matching batch size
        mask = tf.math.logical_and(
            similarity >= (median - std * 2),
            similarity <= 1
        )

        # Ensure mask is properly shaped (batch_size,)
        mask = tf.cast(mask, tf.bool)

        # 🔹 Use tf.boolean_mask() with correctly shaped mask
        filtered_batch = tf.boolean_mask(img_batch, mask)
        filtered_embed = tf.boolean_mask(embed_batch, mask)
        filtered_target = tf.boolean_mask(y_target, mask)

        num_kept = tf.shape(filtered_batch)[0]  # Count how many records will be written
        num_excluded = tf.shape(img_batch)[0] - num_kept  # Remaining records are excluded

        total_written += num_kept.numpy()
        total_excluded += num_excluded.numpy()

        if num_kept > 0:
            for i in range(num_kept.numpy()):
                feature = {
                    'img_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(filtered_batch[i]).numpy()])),
                    'embed_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(filtered_embed[i]).numpy()])),
                    'Y_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(filtered_target[i]).numpy()]))
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())

        # 🔹 Update the progress bar dynamically
        progress_bar.set_postfix({
            "Written": total_written,
            "Excluded": total_excluded
        }, refresh=True)

    writer.close()
    tf.print("Filtered dataset written to", output_tfrecord)
    tf.print(f"Total written: {total_written}, Total excluded: {total_excluded}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with specified parameters.")
    
    parser.add_argument(
        "--tfrecord_shard_path",
        type=str,
        default="./output/kitchensink_256/",
        help="Path to the dataset (default: './data/kitchensink_128/')"
    )

    parser.add_argument(
        "--tfrecord_output_path",
        type=str,
        default=None,
        help="Path to save the filtered dataset (default: None)"
    )
    
    args = parser.parse_args()

    dataset_analyzer = DatasetAnalyzer()

    dataset = tf.data.TFRecordDataset(args.tfrecord_shard_path)
    dataset = dataset.map(lambda x: _parse_function(x, (1,256,256,3), (1,512), (1,256,256,3)), 
                        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(256).prefetch(tf.data.AUTOTUNE)

    if args.tfrecord_output_path:
        #cutoff = dataset_analyzer.analyze_dataset(dataset, progress_every_n=100)
        cutoff=0.4988
        filter_dataset(dataset, args.tfrecord_output_path, dataset_analyzer)
    else:
        dataset_analyzer.analyze_dataset(dataset, progress_every_n=100)


#Example Usage:
#python3 dataset_analysis.py --tfrecord_shard_path=./output/kitchensink_256/combined_shard_0.tfrecord --tfrecord_output_path=./data/kitchensink_quality_filtered_256/combined_filtered_shard_0.tfrecord