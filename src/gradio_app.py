#!/usr/bin/python3

import argparse
from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.55)

import os
import gradio as gr
import cv2
import numpy as np  # Still used for I/O and some operations
from PIL import Image
from inference import AEINETSwapper
import tensorflow as tf
from preprocess_inswapper import do_inswapper_pretraining
from data_loader import get_training_data
import shutil
from tensorflow.keras import mixed_precision
import uuid
import subprocess

mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = 256
#DEFAULT_MODEL = "/app/models/MODEL_256x256_SUPER_v11_BLOCKS2_latest"
#DEFAULT_MODEL = "/app/models/MODEL_256x256_SUPER_v13_BLOCKS2_latest"
DEFAULT_MODEL = "/app/models/MODEL_256x256_SUPER_v14_BLOCKS2_latest"
#DEFAULT_MODEL = "/app/models/MODEL_ID_ONLY_3_BLOCKS2_latest"
#DEFAULT_MODEL = "/app/models/MODEL_256x256_SUPER_ID_ONLY_v1_BLOCKS2_latest"

UPSAMPLE = False
WITH_SUPER_RESOLUTION = True

if WITH_SUPER_RESOLUTION:
    IMAGE_SIZE = 512

assert "libx264" in subprocess.check_output(
    ["ffmpeg", "-v", "quiet", "-encoders"], text=True
), "Your ffmpeg was built without libx264."

def resize_preserving_aspect_ratio_batch(images, max_dim=None):
    """
    Accepts a batch of images (tensor of shape [batch, height, width, channels]) 
    and resizes each one using TensorFlow operations while preserving aspect ratio.
    """
    def process_image(image):
        shape = tf.shape(image)
        h = shape[0]
        w = shape[1]

        if max_dim:
            condition = tf.logical_and(tf.less_equal(w, max_dim), tf.less_equal(h, max_dim))
        else:
            condition = tf.logical_and(True, True)
        
        def process_small():
            # Even if small, perform normalization consistently.
            image_rgb = tf.reverse(image, axis=[-1])  # BGR -> RGB
            image_float = tf.image.convert_image_dtype(image_rgb, tf.float32)
            image_std = tf.image.per_image_standardization(image_float)
            min_val = tf.reduce_min(image_std)
            max_val = tf.reduce_max(image_std)
            image_rescaled = (image_std - min_val) / (max_val - min_val)
            image_uint8 = tf.cast(image_rescaled * 255, tf.uint8)
            return tf.reverse(image_uint8, axis=[-1])  # Back to BGR
        
        def process_large():
            scale = tf.cast(max_dim, tf.float32) / tf.cast(tf.maximum(h, w), tf.float32)
            new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
            new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
            image_rgb = tf.reverse(image, axis=[-1])
            image_float = tf.image.convert_image_dtype(image_rgb, tf.float32)
            resized_float = tf.image.resize(image_float, [new_h, new_w], method=tf.image.ResizeMethod.AREA)
            image_std = tf.image.per_image_standardization(resized_float)
            min_val = tf.reduce_min(image_std)
            max_val = tf.reduce_max(image_std)
            image_rescaled = (image_std - min_val) / (max_val - min_val)
            image_uint8 = tf.cast(image_rescaled * 255, tf.uint8)
            return tf.reverse(image_uint8, axis=[-1])
        
        return tf.cond(condition, process_small, process_large)
    
    return tf.map_fn(process_image, images, dtype=tf.uint8)

def resize_preserving_aspect_ratio_single(image, max_dim=720):
    """
    Resizes a single image by using the batch function.
    """
    image_batch = tf.expand_dims(image, axis=0)
    resized = resize_preserving_aspect_ratio_batch(image_batch, max_dim)
    return tf.squeeze(resized, axis=0)

def process_video_with_image(image_paths, video_path, output_path="/tmp/", batch_size=32, max_dim=None):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    source_images = []
    for image_path in image_paths:
        _, source_image = face_swapper.read_and_scale(image_path, IMAGE_SIZE, swapRGB=False)
        source_image = cv2.copyMakeBorder(
            source_image, 
            50, 50, 50, 50, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        source_images.append(source_image)

    source_face = face_swapper.get_average_face(source_images)

    try:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        ret, first_frame = video.read()
        if not ret:
            print("Could not read even the first frame. Exiting.")
            video.release()
            return
        
        # Process the first frame individually to set output dimensions.
        processed_first_frame = resize_preserving_aspect_ratio_single(first_frame, max_dim=max_dim)
        new_height, new_width = processed_first_frame.shape[:2]
        print(f"New video resolution: {new_width}x{new_height}, FPS: {fps}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = os.path.join(output_path, f"output_{os.path.basename(image_path)}.mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))
    
        # Start with raw frames (without per-frame resizing)
        frame_buffer = [first_frame]
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                if frame_buffer:
                    # Batch resize all raw frames and convert to NumPy before processing.
                    batch_tensor = tf.convert_to_tensor(np.array(frame_buffer))
                    resized_batch = resize_preserving_aspect_ratio_batch(batch_tensor, max_dim=max_dim)
                    processed_frames = face_swapper.process_batch(
                        resized_batch.numpy(), source_face, upsample=UPSAMPLE
                    )
                    for processed_frame in processed_frames:
                        frame_count += 1
                        out.write(
                            processed_frame.numpy() if not isinstance(processed_frame, np.ndarray)
                            else processed_frame
                        )
                break

            frame_buffer.append(frame)

            if len(frame_buffer) == batch_size:
                batch_tensor = tf.convert_to_tensor(np.array(frame_buffer))
                resized_batch = resize_preserving_aspect_ratio_batch(batch_tensor, max_dim=max_dim)
                processed_frames = face_swapper.process_batch(
                    resized_batch.numpy(), source_face, upsample=UPSAMPLE
                )
                for processed_frame in processed_frames:
                    frame_count += 1
                    out.write(
                        processed_frame.numpy() if not isinstance(processed_frame, np.ndarray)
                        else processed_frame
                    )
                frame_buffer = []
    finally:
        video.release()
        out.release()
        print(f"Finished processing {frame_count} frames")
        print(f"Output saved to {output_file}")

    output_file = fix_mp4_for_web(output_file)
    output_file = add_original_audio(output_file, video_path, output_path=output_path)
    
    return output_file

def fix_mp4_for_web(output_file):
    fixed = output_file[:-4] + "_h264.mp4"                                # new name
    subprocess.run(["ffmpeg", "-y", "-i", output_file,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    fixed], check=True)
    return fixed

def add_original_audio(new_video_path, original_video_path, output_path="/tmp/"):
    """
    Extracts the audio from the original video and muxes it with the newly created video.
    The resulting video (with audio) is saved in the output_path.
    """
    new_basename = os.path.basename(new_video_path)
    final_output_file = os.path.join(output_path, f"final_{new_basename}")
    
    command = [
        "ffmpeg",
        "-y",                      # Overwrite output file if it exists
        "-i", new_video_path,      # Input new video (video only)
        "-i", original_video_path, # Input original video (audio source)
        "-c:v", "copy",            # Copy video stream without re-encoding
        "-c:a", "copy",            # Copy audio stream without re-encoding
        "-map", "0:v:0",           # Use video from the first input
        "-map", "1:a:0",           # Use audio from the second input
        final_output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Audio added successfully. Final video saved to {final_output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during audio merging:", e)
        return new_video_path  # Return the new video if audio merging fails
    
    return final_output_file


def create_jit_dataset(image_path, video_path, min_images=320):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    uid = uuid.uuid4().hex
    save_dir = os.path.join("/tmp", uid)
    os.makedirs(save_dir, exist_ok=True)
    
    img_count = 0
    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1
        
        faces = face_swapper.face_analyser.get(frame_bgr)
        if not faces:
            continue

        face = faces[0]
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            expand = 50
            x1_expanded = x1 - expand
            y1_expanded = y1 - expand
            x2_expanded = x2 + expand
            y2_expanded = y2 + expand

            expanded_w = x2_expanded - x1_expanded
            expanded_h = y2_expanded - y1_expanded
            if expanded_w < 1 or expanded_h < 1:
                continue

            expanded_crop_np = np.zeros((expanded_h, expanded_w, 3), dtype=frame_bgr.dtype)
            frame_h, frame_w = frame_bgr.shape[:2]
            src_x1 = max(0, x1_expanded)
            src_y1 = max(0, y1_expanded)
            src_x2 = min(frame_w, x2_expanded)
            src_y2 = min(frame_h, y2_expanded)
            dst_x1 = src_x1 - x1_expanded
            dst_y1 = src_y1 - y1_expanded
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            expanded_crop_np[dst_y1:dst_y2, dst_x1:dst_x2] = frame_bgr[src_y1:src_y2, src_x1:src_x2]
            
            # Resize and normalize using TensorFlow ops
            face_crop = tf.image.resize(expanded_crop_np, (256, 256), method=tf.image.ResizeMethod.AREA)
            face_crop_rgb = tf.reverse(face_crop, axis=[-1])
            face_crop_float = tf.image.convert_image_dtype(face_crop_rgb, tf.float32)
            face_crop_std = tf.image.per_image_standardization(face_crop_float)
            min_val = tf.reduce_min(face_crop_std)
            max_val = tf.reduce_max(face_crop_std)
            face_crop_rescaled = (face_crop_std - min_val) / (max_val - min_val)
            face_crop_uint8 = tf.cast(face_crop_rescaled * 255, tf.uint8)
            face_crop_bgr = tf.reverse(face_crop_uint8, axis=[-1]).numpy()

            output_path = os.path.join(save_dir, f"{video_name_no_ext}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, face_crop_bgr)
            img_count += 1
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()

    if img_count < min_images:
        images = sorted([f for f in os.listdir(save_dir) if f.lower().endswith('.jpg')])
        num_saved = img_count
        idx = 0
        while num_saved < min_images and images:
            src_file = os.path.join(save_dir, images[idx])
            new_filename = f"{video_name_no_ext}_duplicate_{num_saved}.jpg"
            dst_file = os.path.join(save_dir, new_filename)
            shutil.copy2(src_file, dst_file)
            num_saved += 1
            idx = (idx + 1) % len(images)
        print(f"Duplicated images to reach min_images. Now total images: {num_saved}")
        img_count = num_saved

    batch_size = 32
    dataset_path = do_inswapper_pretraining(
        data_dirs=[save_dir],
        output_dir=save_dir,
        use_fixed_image=True,
        fixed_img_from_path=image_path,
        batch_size_override=batch_size
    )
    train, validation = get_training_data(batch_size=batch_size, tfrecord_shard_path=save_dir, p=0.01)
    return (train, validation)

def train_model_for_video(image_path, video_path):
    train_ds, validation_ds = create_jit_dataset(image_path, video_path)
    combined_dataset = validation_ds.concatenate(train_ds)
    face_swapper.tune_for(combined_dataset, epochs=1)
    tuned_output_path = process_video_with_image(image_path, video_path, output_path="/tmp/", batch_size=32)
    return tuned_output_path

# --------------------------------------------------------------------------- #
#  Wrapper: **multiple** videos  ➔  list of outputs (for gallery)
# --------------------------------------------------------------------------- #
def process_videos_with_image(
    image_paths,
    video_paths,
    output_path = None,
    batch_size = 32,
    max_dim = None,
):
    if not image_paths:
        raise gr.Error("Please upload at least one source image.")
    if isinstance(video_paths, (list, tuple)):
        vids = video_paths
    else:
        vids = [video_paths]
    results: list[str] = []
    for vp in vids:
        output_path = os.path.dirname(vp)
        results.append(
            process_video_with_image(
                image_paths=list(image_paths),
                video_path=vp,
                output_path=output_path,
                batch_size=batch_size,
                max_dim=max_dim,
            )
        )
    return results



# --------------------------------------------------------------------------- #
#  Gradio UI
# --------------------------------------------------------------------------- #
with gr.Blocks() as iface:
    with gr.Tab("Processing"):
        # --- Model reload --------------------------------------------------- #
        gr.Button("Reload Model").click(lambda _: face_swapper.reload(), None, None)

        # --- Inputs --------------------------------------------------------- #
        img_in = gr.File(
            label="Upload Images",
            type="filepath",
            file_types=["image"],
            file_count="multiple",
        )
        vid_in = gr.File(
            label="Upload Videos or GIFs",
            type="filepath",
            file_types=["video", "image"],
            file_count="multiple",
        )

        # --- Upsample toggle ------------------------------------------------ #
        upsample_chk = gr.Checkbox(
            label="Upsample",
            value=False, 
        )

        # --- Outputs -------------------------------------------------------- #
        gallery_out = gr.Gallery(
            label="Processed Videos",
            show_label=True,
            columns=2,
        )

        # --- Processing button --------------------------------------------- #
        def _process_with_upsample(images, videos, upsample):
            global UPSAMPLE
            UPSAMPLE = upsample
            return process_videos_with_image(images, videos)

        gr.Button("Process").click(
            _process_with_upsample,
            inputs=[img_in, vid_in, upsample_chk],
            outputs=gallery_out,
        )

        # --- Fine-tune (uses first video only) ----------------------------- #
        def _tune(images, videos):
            first_vid = videos[0] if isinstance(videos, list) else videos
            if not first_vid:
                raise gr.Error("Please upload at least one video for tuning.")
            return train_model_for_video(images, first_vid)

        gr.Button("Tune Face to First Video").click(
            _tune,
            inputs=[img_in, vid_in],
            outputs=gallery_out,
        )

# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to model weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--use_emap",
        action="store_true",
        default=False,
        help="Enable legacy emap training (default: False)"
    )
    args = parser.parse_args()

    face_swapper = AEINETSwapper(
        model_paths=(args.model_weights, ""),
        with_super_resolution=WITH_SUPER_RESOLUTION,
        use_emap=args.use_emap,
    )

    share_flag = os.getenv("SHARE", "false").lower() == "true"
    iface.launch(server_name="0.0.0.0", server_port=5000, share=share_flag)