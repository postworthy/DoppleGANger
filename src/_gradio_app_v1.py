import argparse
from gpu_memory import limit_gpu_memory 
limit_gpu_memory(0.55)

import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from inference import AEINETSwapper
import tensorflow as tf
from preprocess_inswapper import do_inswapper_pretraining
from data_loader import get_training_data
import shutil
from tensorflow.keras import mixed_precision
import uuid

mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = 256
DEFAULT_MODEL="./models/MODEL_256x256_v11_BLOCKS2_latest"
UPSAMPLE=False

def resize_preserving_aspect_ratio(image, max_dim=720):
    """
    Resizes 'image' so that neither width nor height exceed 'max_dim',
    maintaining the aspect ratio. Returns the resized image.
    """
    (h, w) = image.shape[:2]

    # If already within limits, return as is
    if w <= max_dim and h <= max_dim:
        return image

    # Determine the scale factor (whichever dimension is larger is scaled to 'max_dim')
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize and return
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #Normalize
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized_float = tf.image.convert_image_dtype(resized_rgb, tf.float32)
    resized_std = tf.image.per_image_standardization(resized_float)
    min_val = tf.reduce_min(resized_std)
    max_val = tf.reduce_max(resized_std)
    resized_rescaled = (resized_std - min_val) / (max_val - min_val)
    resized_uint8 = tf.cast(resized_rescaled * 255, tf.uint8).numpy()
    resized = cv2.cvtColor(resized_uint8, cv2.COLOR_RGB2BGR)

    return resized

def process_video_with_image(image_path, video_path, output_path="/tmp/", batch_size=8):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    _, source_image = face_swapper.read_and_scale(image_path, IMAGE_SIZE, swapRGB=False)

    source_image = cv2.copyMakeBorder(
        source_image, 
        50, 50, 50, 50, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )

    try:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        ret, first_frame = video.read()
        if not ret:
            print("Could not read even the first frame. Exiting.")
            video.release()
            return
        
        # Resize it (cap at 720 in width/height)
        first_frame = resize_preserving_aspect_ratio(first_frame)
        # Get the dimensions after resizing
        new_height, new_width = first_frame.shape[:2]
        print(f"New video resolution: {new_width}x{new_height}, FPS: {fps}")

        # --- STEP 2: Create the writer with the *resized* dimensions ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        output_file = os.path.join(output_path, f"output_{os.path.basename(image_path)}.mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))
    
        frame_buffer = [first_frame]
        
        while True:
            ret, frame = video.read()

            if not ret:
                if frame_buffer:
                    # Process any remaining frames in the buffer
                    #frame_buffer = normalize_images(frame_buffer)
                    processed_frames = face_swapper.process_batch(np.array(frame_buffer), source_image, upsample=UPSAMPLE)
                    for processed_frame in processed_frames:
                        out.write(processed_frame)
                break

            frame = resize_preserving_aspect_ratio(frame)
            frame_buffer.append(frame)

            if len(frame_buffer) == batch_size:
                # Process the batch of frames
                #frame_buffer = normalize_images(frame_buffer)
                processed_frames = face_swapper.process_batch(np.array(frame_buffer), source_image, upsample=UPSAMPLE)
                for processed_frame in processed_frames:
                    out.write(processed_frame)
                frame_buffer = []
    finally:
        video.release()
        out.release()
        print(f"Output saved to {output_file}")

    return output_file

def create_jit_dataset(image_path, video_path, min_images=320):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    # 1) Prepare the output directory
    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    id=uuid.uuid4().hex
    save_dir = os.path.join("/tmp", id)
    os.makedirs(save_dir, exist_ok=True)
    
    img_count = 0
    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Detect face(s)
        faces = face_swapper.face_analyser.get(frame_bgr)
        if not faces:
            continue  # No faces => skip

        # For simplicity, just use the first face in each frame
        face = faces[0]

        try:
            x1, y1, x2, y2 = face.bbox
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Expand by 50 px on all sides
            expand = 50
            x1_expanded = x1 - expand
            y1_expanded = y1 - expand
            x2_expanded = x2 + expand
            y2_expanded = y2 + expand

            # Build a black canvas of the expanded size
            expanded_w = x2_expanded - x1_expanded
            expanded_h = y2_expanded - y1_expanded
            # If expanded bbox is invalid, skip
            if expanded_w < 1 or expanded_h < 1:
                continue

            # Create a black background for the expanded region
            expanded_crop = np.zeros((expanded_h, expanded_w, 3), dtype=frame_bgr.dtype)

            # Compute intersection of expanded area with actual frame
            frame_h, frame_w = frame_bgr.shape[:2]
            src_x1 = max(0, x1_expanded)
            src_y1 = max(0, y1_expanded)
            src_x2 = min(frame_w, x2_expanded)
            src_y2 = min(frame_h, y2_expanded)

            # Destination offsets within the black canvas
            dst_x1 = src_x1 - x1_expanded
            dst_y1 = src_y1 - y1_expanded
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            # Copy the overlapping region from the original frame
            expanded_crop[dst_y1:dst_y2, dst_x1:dst_x2] = \
                frame_bgr[src_y1:src_y2, src_x1:src_x2]

            # Now resize to the final desired 256×256
            face_crop = cv2.resize(expanded_crop, (256, 256), interpolation=cv2.INTER_AREA)

            #Normalization
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop_float = tf.image.convert_image_dtype(face_crop_rgb, tf.float32)
            face_crop_std = tf.image.per_image_standardization(face_crop_float)
            min_val = tf.reduce_min(face_crop_std)
            max_val = tf.reduce_max(face_crop_std)
            face_crop_rescaled = (face_crop_std - min_val) / (max_val - min_val)
            face_crop_uint8 = tf.cast(face_crop_rescaled * 255, tf.uint8).numpy()
            face_crop = cv2.cvtColor(face_crop_uint8, cv2.COLOR_RGB2BGR)

            # 2) Save the crop as a JPG
            output_path = os.path.join(save_dir, f"{video_name_no_ext}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, face_crop)
            img_count += 1
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()

    #We need a minimum number for the dataset
    if img_count < min_images:
        # Collect all the filenames we've saved
        images = [f for f in os.listdir(save_dir) if f.lower().endswith('.jpg')]
        images.sort()  # optional, keeps the naming consistent
        num_saved = img_count
        idx = 0

        # Keep copying images in a round-robin fashion until we reach min_images
        while num_saved < min_images and images:
            src_file = os.path.join(save_dir, images[idx])
            # Build a new filename
            new_filename = f"{video_name_no_ext}_duplicate_{num_saved}.jpg"
            dst_file = os.path.join(save_dir, new_filename)
            
            shutil.copy2(src_file, dst_file)
            
            num_saved += 1
            idx = (idx + 1) % len(images)

        print(f"Duplicated images to reach min_images. Now total images: {num_saved}")
        img_count = num_saved  # update the final count

    batch_size=16
    # Depending on your pipeline, proceed with further steps:
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
    (train, validation) = create_jit_dataset(image_path, video_path)
    combined_dataset = validation.concatenate(train)
    face_swapper.tune_for(combined_dataset, epochs=1)

    tuned_output_path = process_video_with_image(image_path, video_path, output_path="/tmp/")
    
    return tuned_output_path


# Define the Gradio interface
with gr.Blocks() as iface:
    # Tab 1: Processing
    with gr.Tab("Processing"):
        #Process
        process_button = gr.Button("Reload Model")
        process_button.click(
            lambda _: face_swapper.reload(),
            inputs=None,
            outputs=None
        )

        image_input = gr.Image(type="filepath", label="Upload Image")
        video_input = gr.File(label="Upload Video or GIF", file_types=["video", "image"])
        output_video = gr.Video(label="Processed Video")
        
        #Process
        process_button = gr.Button("Process")
        process_button.click(
            process_video_with_image,
            inputs=[image_input, video_input],
            outputs=output_video
        )

        # SECOND BUTTON FOR TUNING
        tune_button = gr.Button("Tune Face to Video")
        tune_button.click(
            train_model_for_video,
            inputs=[image_input, video_input],
            outputs=output_video
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start gradio app with specified parameters.")
    
    parser.add_argument(
        "--model_weights",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to the model weights (default: '{DEFAULT_MODEL}')"
    )
    
    args = parser.parse_args()

    face_swapper = AEINETSwapper(model_paths=(args.model_weights, f""))
    
    share = os.environ.get("SHARE", "False").lower() == "true"
    iface.launch(server_name="0.0.0.0", server_port=5000, share=share)
